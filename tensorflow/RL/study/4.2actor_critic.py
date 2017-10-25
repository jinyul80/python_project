import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import time
from collections import deque

from gridworld import gameEnv

env = gameEnv(partial=False, size=5)


# 신경망 구현
class ActorCriticNetwork():
    def __init__(self, output_size, learning_rate):
        self.output_size = output_size
        self.learning_rate = learning_rate

        self._build()

    def _build(self):
        with tf.variable_scope('thread_0'):
            # 신경망은 게임으로부터 벡터화된 배열로 프레임을 받아서
            # 이것을 리사이즈 하고, 4개의 콘볼루션 레이어를 통해 처리한다.

            # 입력값을 받는 부분 21168 차원은 84*84*3 의 차원이다.
            self.state = tf.placeholder(shape=[None, 21168], dtype=tf.float32, name='input_state')
            self.actions = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32, name='input_action')
            self.advantages = tf.placeholder(tf.float32, [None, 1], name="advantages")
            self.rewards = tf.placeholder(tf.float32, [None, 1], name="reward")

            # LSTM 설정 인자
            self.batch_size = tf.placeholder(dtype=tf.int32)
            self.trainLength = tf.placeholder(dtype=tf.int32)

            # conv2d 처리를 위해 84x84x3 으로 다시 리사이즈
            _imageIn = tf.reshape(self.state, shape=[-1, 84, 84, 3])

            # 콘볼루션 레이어
            _conv1 = tf.layers.conv2d(inputs=_imageIn, filters=32, kernel_size=[3, 3], strides=[2, 2],
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      activation=tf.nn.relu)

            _conv2 = tf.layers.conv2d(inputs=_conv1, filters=64, kernel_size=[3, 3], strides=[2, 2],
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      activation=tf.nn.relu)

            _conv3 = tf.layers.conv2d(inputs=_conv2, filters=128, kernel_size=[3, 3], strides=[2, 2],
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      activation=tf.nn.relu)
            # _conv3 = tf.layers.dropout(inputs=_conv3, rate=0.5, training=True)

            # Dense 레이어
            _flat = tf.reshape(_conv3, shape=[-1, 9 * 9 * 128])
            _dense1 = tf.layers.dense(inputs=_flat, units=1024, activation=tf.nn.relu)

            # LSTM
            _rnn_in = tf.expand_dims(_dense1, [0])
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=512, state_is_tuple=True)
            # cell 초기화
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            # 은닉층 초기화
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            # 셀에 넣을 값 받기
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            # 은닉층에 넣을 값 받기
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            # 배치사이즈
            step_size = tf.shape(_imageIn)[:1]
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                inputs=_rnn_in, cell=lstm_cell, initial_state=state_in, sequence_length=step_size,
                time_major=False, dtype=tf.float32)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            # 최종을 다시 벡터화
            rnn_out = tf.reshape(lstm_outputs, [-1, 512])

            # Output
            self.pred = tf.layers.dense(inputs=rnn_out, units=self.output_size, activation=tf.nn.softmax, name='pred')
            self.values = tf.layers.dense(inputs=rnn_out, units=1, name="values")

            # Loss 계산
            _policy_gain = -tf.reduce_sum(tf.log(self.pred + 1e-5) * self.actions * self.advantages, axis=1)
            _policy_gain = tf.reduce_mean(_policy_gain)
            _entropy = - tf.reduce_sum(self.pred * tf.log(self.pred + 1e-5), axis=1)
            _entropy = tf.reduce_mean(_entropy)
            _value_loss = tf.losses.mean_squared_error(self.values, self.rewards, scope="value_loss")

            self.loss = _policy_gain + (_value_loss * 0.5) - (_entropy * 0.01)

            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            tf.summary.scalar("a_pred_max", tf.reduce_mean(tf.reduce_max(self.pred, axis=1)))
            tf.summary.scalar("policy_loss", _policy_gain)
            tf.summary.scalar("entropy_loss", _entropy)
            tf.summary.scalar("value_loss", _value_loss)
            tf.summary.scalar("total_loss", self.loss)
            tf.summary.histogram("values", self.values)
            tf.summary.histogram("pred", self.pred)

            self.reward_avg = tf.placeholder(tf.float32, name="reward_avg")
            tf.summary.scalar("reward_avg", self.reward_avg)

            self.summary_op = tf.summary.merge_all()


class Agent:
    def __init__(self, env, network, action_n, mini_batch_size):
        self.env = env
        self.model = network
        self.sess = tf.get_default_session()
        self.action_n = action_n
        self.mini_batch_size = mini_batch_size

   # Model 학습
    def train(self, buffer, batch_size, frame_size, avg_reward):
        for _ in range(self.mini_batch_size):
            # 경험으로부터 랜덤한 배치를 뽑는다
            xs, ys, train_rewards = self._get_sample_batch(buffer, batch_size, frame_size)

            rnn_state = self.model.state_init

            feed = {
                self.model.state: xs,
                self.model.state_in[0]: rnn_state[0],
                self.model.state_in[1]: rnn_state[1]
            }

            V = self.sess.run(self.model.values, feed_dict=feed)
            ADV = train_rewards - V
            ADV = (ADV - np.mean(ADV)) / (np.std(ADV) + 1e-8)

            feed = {
                self.model.state: xs,
                self.model.actions: ys,
                self.model.advantages: ADV,
                self.model.rewards: train_rewards,
                self.model.reward_avg: avg_reward,
                self.model.state_in[0]: rnn_state[0],
                self.model.state_in[1]: rnn_state[1]
            }

            _, summ = self.sess.run([self.model.train, self.model.summary_op], feed_dict=feed)

        return summ

    # 랜덤한 위치에서 frame size 만큼 덧 붙여서 샘플 추출
    def _get_sample_batch(self, buffer, batch_size, frame_size):
        temp_idx = np.arange(frame_size, len(buffer))
        np.random.shuffle(temp_idx)

        state_list = []
        action_list = []
        reward_list = []

        for i in range(batch_size):
            idx = temp_idx[i]
            for num in reversed(range(frame_size)):
                sub_i = idx - num
                state_list.append(buffer[sub_i][0])
                action_list.append(buffer[sub_i][1])
                reward_list.append(buffer[sub_i][2])

        state_list = np.vstack(state_list)
        action_list = np.vstack(action_list)
        reward_list = np.vstack(reward_list)

        return state_list, action_list, reward_list


# State 전처리
def processState(states):
    return np.reshape(states, [21168])


# 보상 배열을 감쇠 적용
def discount_rewards(r, gamma=0.99):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            # reset the sum, since this was a game boundary (pong specific!)
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    # 노멀라이징 적용
    discounted_r -= np.mean(discounted_r)
    if np.std(discounted_r) != 0:
        discounted_r /= np.std(discounted_r)

    return discounted_r


# 학습 파라미터 설정
batch_size = 16  # 각 학습 단계에 대해 얼마나 많은 경험을 사용할지 결정
mini_batch_steps = 5  # Mini 배치를 실행할 횟수
update_freq = 5  # 학습 단계를 얼마나 자주 수행할 것인가
frame_size = 10  # LSTM 시퀀스 size
num_episodes = 100000  # 몇개의 에피소드를 할 것인가.
max_epAction_count = 50  # 에피소드의 최대 길이 (50 걸음)
max_buffer_size = 5000
load_model = True  # 저장된 모델을 불러올 것인가?
path = "./log/4.2_actor_critic_log"  # 모델을 저장할 위치
gamma = 0.99  # 보상에 대한 할인 인자
learning_rate = 1e-5  # 학습 속도
mode = 'train'  # 작동 모드 설정

if mode == 'test':
    env.show_game = True

# 그래프를 초기화한다
tf.reset_default_graph()
# 텐서플로 세션을 연다
sess = tf.InteractiveSession()

# Define A2C(Actor-Critic) and Agent
ac_network = ActorCriticNetwork(env.actions, learning_rate)
agent = Agent(env, ac_network, env.actions, mini_batch_steps)

# Global step
global_step = tf.Variable(0, name="global_step", trainable=False)

# 변수들을 초기화한다
init = tf.global_variables_initializer()
sess.run(init)

# saver를 만든다
saver = tf.train.Saver()

# Summary
summary_writer = tf.summary.FileWriter(path)
summary_writer.add_graph(sess.graph)

# 경험을 저장할 장소
myBuffer = deque()

# 에피소드별 총 보상을 저장할 리스트를 만든다
score_list = []

# 모델을 세이브할 장소를 만든다.
if not os.path.exists(path):
    os.makedirs(path)

# Start episode no
start_step = 1

# 모델을 불러올지 체크
if load_model == True:
    # 모델을 불러온다
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt != None:
        saver.restore(sess, ckpt.model_checkpoint_path)
        start_step = sess.run(global_step) + 1
        print("Successfully loaded:", ckpt.model_checkpoint_path)
        print('Start step :', start_step)

# Set episode start time
start_time = time.time()

# 에피소드 시작
for episode in range(start_step, num_episodes):

    # Set global step
    sess.run(tf.assign(global_step, episode))

    # 환경과 처음 상태을 초기화한다
    s = env.reset()

    # 종료 여부
    done = False
    # 보상
    episode_reward = 0
    # 걸음
    action_count = 0
    # 에피스드 임시 버퍼
    episode_buffer = []

    rnn_state = agent.model.state_init

    # Episode 기록
    while not done:
        # 만약 50 걸음보다 더 간다면 종료한다.
        if action_count >= max_epAction_count:
            break

        action_count += 1

        s = processState(s)

        # Action 추출
        feed = {
            agent.model.state: [s],
            agent.model.state_in[0]: rnn_state[0],
            agent.model.state_in[1]: rnn_state[1]
        }
        action_prob, rnn_state = agent.sess.run([agent.model.pred, agent.model.state_out],
                                                    feed_dict=feed)
        action_prob = np.squeeze(action_prob)
        a = np.random.choice(env.actions, size=1, p=action_prob)[0]

        # 주어진 행동을 실행하고 다음 상태, 보상, 종료 여부를 가져옴
        s1, reward, done = env.step(a)

        # Action에 one_hot 적용
        action_one_hot = np.squeeze(np.eye(env.actions)[a:a + 1])

        # 버퍼에 현재 상태, 행동, 보상, 다음 상태, 종료 여부를 저장한다
        episode_buffer.append([s, action_one_hot, reward])

        # 총 보상
        episode_reward += reward

        # 상태를 바꾼다.
        s = s1

    if mode == 'test':
        print('Test score : {}'.format(episode_reward))
        break

    elif mode == 'train':
        # 보상을 저장한다
        score_list.append(episode_reward)

        # 점수를 감쇠 적용 후 버퍼에 에피소드 행동들 저장
        dis_rewards = np.reshape(episode_buffer, [len(episode_buffer), 3])[:, 2]
        dis_rewards = discount_rewards(dis_rewards)
        for idx in range(len(episode_buffer)):
            temp_ep = episode_buffer[idx]
            temp_x = temp_ep[0]
            temp_y = temp_ep[1]
            temp_reward = dis_rewards[idx]

            myBuffer.append([temp_x, temp_y, temp_reward])

            # Max 버퍼 사이즈 보다 클 경우 오래된 항목 삭제
            if len(myBuffer) > max_buffer_size:
                myBuffer.popleft()

        # 일정한 episode 마다 학습 실행
        if episode % update_freq == 0:
            # Train
            summ = agent.train(myBuffer, batch_size, frame_size, np.mean(score_list[-100:]))
            summary_writer.add_summary(summ, episode)

            # 러닝 시간
            duration = time.time() - start_time
            sec_per_step = float(duration)

            # 시작 시간 초기화
            start_time = time.time()

            # 최근 에피소드의 평균 점수값을 나타낸다.
            print('episode : {}, Average: {:.2f}, Avg100: {:.2f} ({:.3f} sec)'.format(episode,
                                                                                   np.mean(score_list[-update_freq:]),
                                                                                   np.mean(score_list[-100:]),
                                                                                   sec_per_step))

        # 주기적으로 모델을 저장한다
        if episode % 1000 == 0:
            saver.save(sess, os.path.join(path, "model.ckpt"))
            print("Saved Model")

if mode == 'train':
    # 모델을 저장한다.
    saver.save(sess, os.path.join(path, "model.ckpt"))

    # 성공확률을 표시
    print("Percent of succesful episodes: " + str(sum(score_list) / num_episodes) + "%")

    # Pyplot 초기화 및 score 로그 출력
    plt.close()
    plt.title('Policy gradient')
    plt.plot(range(len(score_list)), score_list, color='blue')
    plt.show()

# 텐서플로 세션을 종료한다
sess.close()