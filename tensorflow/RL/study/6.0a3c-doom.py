import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from helper import *
from vizdoom import *

from random import choice
from time import sleep
from time import time

# 학습 변수들을 다른데에 복사
# 워커 신경망의 파라미터를 전역 신경망의 파라미터로 설정
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Doom 화면을 자르고 리사이즈 하는 처리
def process_frame(frame):
    s = frame[10:-10,30:-30]
    s = scipy.misc.imresize(s,[84,84])
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s

# 할인된 보상을 계산하는 함수
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# 정책과 가치 출력 레이어에 대한 가중치를 초기화
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            # 입력 값을 받는 부분
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            # 이미지로 리사이즈
            self.imageIn = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])
            # 콘볼루션을 통해 이미지 인코딩
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.imageIn, num_outputs=16,
                                     kernel_size=[8, 8], stride=[4, 4], padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv1, num_outputs=32,
                                     kernel_size=[4, 4], stride=[2, 2], padding='VALID')
            # FC로 256차원으로 만듦
            hidden = slim.fully_connected(slim.flatten(self.conv2), 256, activation_fn=tf.nn.elu)

            # 시간 의존성에 대한 순환 신경망
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)
            # cell 과 hidden state를 정의
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            # cell 과 hidden state 에 대해 받는 부분
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            # lstm 으로 상태를 계산
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            # RNN 출력
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # RNN 출력값으로 정책과 가치를 계산
            self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # 워커 신경망에만 손실함수와 그라디언트를 업데이트하는 연산이 필요함
            if scope != 'global':
                # 행동을 받음
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                # 행동을 one hot encoding 함
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                # 타겟 가치를 받음 (할인된 보상)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                # 할인된 보상을 받음
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                # 각 행동에 대한 값을 계산
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # 손실 함수
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # 지역 손실을 이용해 지역별 그라디언트를 구함
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # 전역 신경망에 적용
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


class Worker():
    def __init__(self, game, name, s_size, a_size, trainer, model_path, global_episodes):
        # 워커 이름
        self.name = "worker_" + str(name)
        # 숫자
        self.number = name
        # 모델 경로
        self.model_path = model_path
        # 학습기 adma
        self.trainer = trainer
        # 에피소드 수
        self.global_episodes = global_episodes
        # 증가
        self.increment = self.global_episodes.assign_add(1)
        # 에피소드 보상
        self.episode_rewards = []
        # 에피소드 길이
        self.episode_lengths = []
        # 에피소드 평균값
        self.episode_mean_values = []
        # 기록
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        # 지역 신경망의 파라미터들을 전역 신경망과 동일하게 설정
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        # Doom 환경 설졍
        game.set_doom_scenario_path("basic.wad")  # This corresponds to the simple task we will pose our agent
        game.set_doom_map("map01")
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.add_available_button(Button.MOVE_LEFT)
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.ATTACK)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(Mode.PLAYER)
        game.init()
        self.actions = [[True, False, False], [False, True, False], [False, False, True]]
        # End Doom set-up
        self.env = game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        # 상태
        observations = rollout[:, 0]
        # 행동
        actions = rollout[:, 1]
        # 보상
        rewards = rollout[:, 2]
        # 다음 상태
        next_observations = rollout[:, 3]
        # 가치
        values = rollout[:, 5]

        # 경험으로부터 보상과 가치를 받아서, 이를 이용해 이득과 할인된 보상을 만든다.
        # 이 이득 함수는 "일반화 이득 추정 Generlized Advantage Estimation"을 사용한다

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # 손실로 만든 그라디언트로 전역 신경망을 업데이트한다.
        # 신경망에 관한 통계량을 주기적으로 저장한다.
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, sess, coord, saver):
        # 에피소드 수
        episode_count = sess.run(self.global_episodes)
        # 총 단계
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                self.env.new_episode()
                s = self.env.get_state().screen_buffer
                episode_frames.append(s)
                s = process_frame(s)
                rnn_state = self.local_AC.state_init

                while self.env.is_episode_finished() == False:
                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs: [s],
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    r = self.env.make_action(self.actions[a]) / 100.0
                    d = self.env.is_episode_finished()
                    if d == False:
                        s1 = self.env.get_state().screen_buffer
                        episode_frames.append(s1)
                        s1 = process_frame(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # 에피소드가 끝나지 않다하더라도, 경험 버퍼가 가득찬다면, 경험 버퍼를 이용해 업데이트를 한다.
                    if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length - 1:
                        # 우리는 실제 마지막 보상값이 무엇인지 알지 못하기 때문에, 우리는 우리의 가치 추정으로부터 "부트스트랩"한다.
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.inputs: [s],
                                                 self.local_AC.state_in[0]: rnn_state[0],
                                                 self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
                        # 손실
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break
                # 에피소드 보상
                self.episode_rewards.append(episode_reward)
                # 에피소드 길이
                self.episode_lengths.append(episode_step_count)
                # 에피소드 평균 가치
                self.episode_mean_values.append(np.mean(episode_values))

                # 에피소드의 끝에ㅓㅅ 경험 버퍼를 사용해 신경망을 업데이트
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # 주기적으로 에피소드의 gif 를 저장하고, 모델 파라미터와 요약 통계량을 저장한다.
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(images, './frames/image' + str(episode_count) + '.gif',
                                 duration=len(images) * time_per_step, true_image=True, salience=False)
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

max_episode_length = 1
gamma = .99 # 할인율
s_size = 7056 # 상태공간 84 * 84 * 1 의 그레이스케일 이미지
a_size = 3 # 행동공간 좌, 우, 총쏘기
load_model = False
model_path = './model'

# 텐서플로 초기화
tf.reset_default_graph()

# 모델 경로가 없으면 만들어줌
if not os.path.exists(model_path):
    os.makedirs(model_path)

# gif 파일이 저장될 폴더 만들어줌
if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    # 전역 신경망
    master_network = AC_Network(s_size, a_size, 'global', None)
    # 가능한 cpu 스레드 수만큼 워커를 설정
    num_workers = multiprocessing.cpu_count()
    workers = []
    # 워커를 인스턴트해서 모음
    for i in range(num_workers):
        workers.append(Worker(DoomGame(), i, s_size, a_size, trainer, model_path, global_episodes))
    # 모델 저장
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    # 스레드  관리자
    coord = tf.train.Coordinator()
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # 이부분이 비동기 마법 부분
    # 각 워커들은 서로다른 스레드에서 시작
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)