import gym
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 신경망 구현
class Model():
    def __init__(self, output_size, learning_rate):
        self._output_size = output_size
        self._learning_rate = learning_rate

        self._build()

    def _build(self):
        with tf.variable_scope('cnn'):
            # 신경망은 게임으로부터 벡터화된 배열로 프레임을 받아서
            # 이것을 리사이즈 하고, 4개의 콘볼루션 레이어를 통해 처리한다.

            # 입력값을 받는 부분 21168 차원은 84*84*3 의 차원이다.
            self._X = tf.placeholder(shape=[None, 21168], dtype=tf.float32, name='input_x')

            self._Y = tf.placeholder(shape=[None, self._output_size], dtype=tf.float32, name='input_y')
            self._advantages = tf.placeholder(tf.float32, [None, 1], name="reward_signal")

            self.batch_size = tf.placeholder(dtype=tf.int32)
            self.trainLength = tf.placeholder(dtype=tf.int32)

            # conv2d 처리를 위해 84x84x3 으로 다시 리사이즈
            _imageIn = tf.reshape(self._X, shape=[-1, 84, 84, 3])

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
            _conv3 = tf.layers.dropout(inputs=_conv3, rate=0.8, training=True)

            # Dense 레이어
            _flat = tf.reshape(_conv3, shape=[-1, 9 * 9 * 128])
            _dense1 = tf.layers.dense(inputs=_flat, units=1024, activation=tf.nn.relu)
            # self._pred = tf.layers.dense(inputs=_dense1, units=self._output_size, activation=tf.nn.softmax)

            _rnn_in = tf.reshape(_dense1, [self.batch_size, self.trainLength, 1024])
            cell = tf.nn.rnn_cell.LSTMCell(num_units=512, state_is_tuple=True, activation=tf.nn.tanh)

            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
                inputs=_rnn_in, cell=cell, dtype=tf.float32)

            self.rnn = tf.reshape(self.rnn, [-1, 512])
            self._pred = tf.layers.dense(inputs=self.rnn, units=self._output_size, activation=tf.nn.softmax)

            _action_pred = tf.clip_by_value(self._pred, 1e-10, 0.99)

            # self._log_lik = -self._Y * tf.log(_action_pred)
            # self._loss = tf.reduce_mean(tf.reduce_sum(self._log_lik * self._advantages, axis=1))

            self._loss = -tf.reduce_sum(tf.log(_action_pred) * self._Y * self._advantages)

            self._train = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._loss)

    def get_action(self, sess, X, batch_size, length):
        return sess.run(self._pred, feed_dict={self._X: X, self.batch_size: batch_size, self.trainLength: length})

    def train(self, sess, X, Y, rewards, batch_size, length):
        return sess.run([self._train, self._loss], feed_dict={self._X: X, self._Y: Y,
                                                              self._advantages: rewards,
                                                              self.batch_size: batch_size,
                                                              self.trainLength: length})


def processState(states):
    return np.reshape(states, [21168])


def discount_rewards(r, gamma=0.99):
    # 보상 배열을 감쇠 적용
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            # reset the sum, since this was a game boundary (pong specific!)
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    # 노벌라이징 적용
    discounted_r -= np.mean(discounted_r)
    if np.std(discounted_r) != 0:
        discounted_r /= np.std(discounted_r)

    return discounted_r


def get_sample_batch(buffer, batch_size, frame_size):
    temp_idx = np.arange(frame_size, len(buffer))
    np.random.shuffle(temp_idx)

    state_list = []
    action_list = []
    reward_list = []

    temp_buffer = np.reshape(buffer, [len(buffer), 3])

    for i in range(batch_size):
        idx = temp_idx[i]
        for sub_i in reversed(range(frame_size)):
            state_list.append(temp_buffer[idx - sub_i, 0])
            action_list.append(temp_buffer[idx - sub_i, 1])
            reward_list.append(temp_buffer[idx - sub_i, 2])

    state_list = np.vstack(state_list)
    action_list = np.vstack(action_list)
    reward_list = np.vstack(reward_list)

    return state_list, action_list, reward_list


# 학습 파라미터 설정
batch_size = 16  # 각 학습 단계에 대해 얼마나 많은 경험을 사용할지 결정
mini_batch_steps = 10 # Mini 배치를 실행할 횟수
update_freq = 5  # 학습 단계를 얼마나 자주 수행할 것인가
frame_size = 8 # LSTM 시퀀스 size
num_episodes = 1000000  # 몇개의 에피소드를 할 것인가.
pre_train_episodes = 20  # 학습 시작 전에 몇번의 무작위 episode 를 할 것인가.
max_epAction_count = 50  # 에피소드의 최대 길이 (50 걸음)
load_model = True  # 저장된 모델을 불러올 것인가?
path = "./4.1policy_log"  # 모델을 저장할 위치
gamma = 0.99  # 보상에 대한 할인 인자
learning_rate = 1e-3  # 학습 속도

# 그래프를 초기화한다
tf.reset_default_graph()
# 주요 신경망을 만든다
mainQN = Model(env.actions, learning_rate)

# Summray 관련 변수 설정
summary_reward = tf.placeholder(tf.float32, shape=(), name="reward")
summray_loss = tf.placeholder(tf.float32, shape=(), name="loss")
tf.summary.scalar('reward', summary_reward)
tf.summary.scalar('loss', summray_loss)
summary_op = tf.summary.merge_all()

# Global step
global_step = tf.Variable(0, name="global_step", trainable=False)

# 변수들을 초기화한다
init = tf.global_variables_initializer()

# saver를 만든다
saver = tf.train.Saver()

# 학습가능한 변수를 꺼낸다
trainables = tf.trainable_variables()

# 경험을 저장할 장소
myBuffer = deque()

# 에피소드별 총 보상과 걸음을 저장할 리스트를 만든다
jList = []
rList = []

# 모델을 세이브할 장소를 만든다.
if not os.path.exists(path):
    os.makedirs(path)

# 텐서플로 세션을 연다
sess = tf.InteractiveSession()

# 변수를 초기화한다.
sess.run(init)

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

# Summary writer 생성
writer = tf.summary.FileWriter(path)
writer.add_graph(sess.graph)

# Set episode start time
start_time = time.time()

# 에피소드 시작
for episode in range(start_step, num_episodes):

    # Set global step
    sess.run(tf.assign(global_step, episode))

    # 환경과 처음 상태을 초기화한다
    s = env.reset()
    s = processState(s)
    # 종료 여부
    done = False
    # 보상
    episode_reward = 0
    # 걸음
    action_count = 0

    episode_buffer = []

    # Episode
    while not done:
        # 만약 50 걸음보다 더 간다면 종료한다.
        if action_count >= max_epAction_count:
            break

        action_count += 1
        # Q-network 로부터 행동을 greedy 하게 선택하거나 e의 확률로 무작위 행동을 한다
        if episode < pre_train_episodes:
            a = np.random.randint(0, env.actions)
        else:
            # 신경망을 통해 Q 값을 가져오는 부분
            action_prob = mainQN.get_action(sess, [s], 1, 1)
            action_prob = np.squeeze(action_prob)
            a = np.random.choice(env.actions, size=1, p=action_prob)[0]

        # 주어진 행동을 실행하고 다음 상태, 보상, 종료 여부를 가져옴
        s1, reward, done = env.step(a)
        # 상태를 다시 21168 차원으로 리사이즈
        s1 = processState(s1)

        # Action에 one_hot 적용
        action_one_hot = np.squeeze(np.eye(env.actions)[a:a + 1])

        # 버퍼에 현재 상태, 행동, 보상, 다음 상태, 종료 여부를 저장한다
        episode_buffer.append([s, action_one_hot, reward, s1, done])

        # 총 보상
        episode_reward += reward
        # 상태를 바꾼다.
        s = s1

    # 걸음을 저장한다.
    jList.append(action_count)
    # 보상을 저장한다
    rList.append(episode_reward)

    # 버퍼에 에피소드 행동들 저장
    dis_rewards = np.reshape(episode_buffer, [len(episode_buffer), 5])[:, 2]
    dis_rewards = discount_rewards(dis_rewards)

    for idx in range(len(episode_buffer)):
        temp_ep = episode_buffer[idx]
        temp_x = temp_ep[0]
        temp_y = temp_ep[1]
        temp_reward = dis_rewards[idx]

        myBuffer.append([temp_x, temp_y, temp_reward])
        if len(myBuffer) > 1000:
            myBuffer.popleft()

    # 무작위 행동의 수를 넘으면 시작
    if episode > pre_train_episodes:
        # 일정 episode 마다 업데이트 시작
        if episode % update_freq == 0:

            loss_list = []

            for _ in range(mini_batch_steps):
                # 경험으로부터 랜덤한 배치를 뽑는다
                xs, ys, train_rewards = get_sample_batch(myBuffer, batch_size, frame_size)

                _, tLoss = mainQN.train(sess, xs, ys, train_rewards, batch_size, frame_size)
                loss_list.append(tLoss)

            # Summary 기록
            ss = sess.run(summary_op, feed_dict={summray_loss: np.mean(loss_list),
                                                 summary_reward: np.mean(rList[-update_freq:])})
            writer.add_summary(ss, episode)
            
            # 러닝 시간
            duration = time.time() - start_time
            sec_per_step = float(duration)
            
            # 시작 시간 초기화
            start_time = time.time()

            # 최근 에피소드의 평균 점수값을 나타낸다.
            print('episode : {} Average : {:.2f} Loss : {:.6f} ({:.3f} sec)'.format(episode,
                                                                                    np.mean(rList[-update_freq:]),
                                                                                    np.mean(loss_list),
                                                                                    sec_per_step))

            test_pause = '1'

    # 주기적으로 모델을 저장한다
    if episode % 1000 == 0:
        saver.save(sess, os.path.join(path, "model.ckpt"))
        print("Saved Model")

# 모델을 저장한다.
saver.save(sess, os.path.join(path, "model.ckpt"))

# 텐서플로 세션을 종료한다
sess.close()

# 성공확률을 표시
print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")
