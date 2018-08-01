import os
# import pylab
import matplotlib.pyplot as pylab
import numpy as np
from keras.layers import Dense, Input, LSTM, TimeDistributed, Reshape, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import threading
import argparse
import time
from collections import deque

import mini_pacman

# np.random.seed(777)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description="Simple 'argparse' demo application")
parser.add_argument('--mode', default='train', help='Execute mode')
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--max_steps', default=5000, type=int)
parser.add_argument('--n_threads', default=4, type=int)

args = parser.parse_args()


# 신경망 에이전트
class Agent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # 하이퍼파라미터
        self.actor_lr = args.learning_rate
        self.critic_lr = args.learning_rate

        # 정책신경망과 가치신경망 생성
        self.actor, self.critic = self.build_model()
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

    # 신경망 생성
    def build_model(self):
        input = Input(shape=[self.state_size, ])
        net = Dense(128, activation='relu')(input)
        net = Dense(64, activation='relu')(net)

        net = Reshape((1, 64))(net)
        net = LSTM(32, activation='relu')(net)

        net = BatchNormalization()(net)

        policy = Dense(self.action_size, activation='softmax')(net)
        value = Dense(1, activation='linear')(net)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        # 가치와 정책을 예측하는 함수를 만들어냄
        actor._make_predict_function()
        critic._make_predict_function()

        return actor, critic

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        # 정책 크로스 엔트로피 오류함수
        action_prob = -K.sum(action * K.log(policy + 1e-5), axis=1) * advantages
        cross_entropy = K.mean(action_prob)

        # 탐색을 지속적으로 하기 위한 엔트로피 오류
        entropy = -K.sum(policy * K.log(policy + 1e-5), axis=1)
        entropy = K.mean(entropy)

        # 두 오류함수를 더해 최종 오류함수를 만듬
        loss = cross_entropy - (entropy * 0.01)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        # [반환값 - 가치]의 제곱을 오류함수로 함
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_prediction],
                           [loss], updates=updates)
        return train

    # 정책으로 행동 선택
    def get_action(self, state):
        action = self.actor.predict(state)[0]
        return np.random.choice(self.action_size, size=1, p=action)[0]


class RunnerThread(threading.Thread):
    stop_flag = False
    global_episode = 1
    recently_avg_list = deque(maxlen=100)
    all_avg_list = []

    def __init__(self, id, state_size, action_size, global_network, is_training=True):
        super(RunnerThread, self).__init__()

        self.is_training = is_training
        self.render = not(is_training)
        self.env = mini_pacman.Gym(show_game=self.render)


        self.id = id

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        self.global_network = global_network
        self.local_network = Agent(self.state_size, self.action_size)

        self.local_network.actor.set_weights(self.global_network.actor.get_weights())
        self.local_network.critic.set_weights(self.global_network.critic.get_weights())

    def run(self):
        while not RunnerThread.stop_flag:
            self.play_episode()

    # 로컬신경망을 글로벌신경망으로 업데이트
    def update_local_model(self):
        self.local_network.actor.set_weights(self.global_network.actor.get_weights())
        self.local_network.critic.set_weights(self.global_network.critic.get_weights())

    def play_episode(self):

        local_episode = RunnerThread.global_episode

        if local_episode >= args.max_steps:
            RunnerThread.stop_flag = True
            return

        RunnerThread.global_episode += 1

        if self.render:
            self.env.show_game = True

        # Set episode start time
        start_time = time.time()

        self.update_local_model()

        done = False
        score = 0

        xs, ys, reward_list, value_list = [], [], [], []

        # env 초기화
        state = self.env.reset()

        while not done:
            # 현재 상태로 행동을 선택
            st = np.reshape(state, [1, self.state_size])

            action = self.local_network.get_action(st)
            val = self.local_network.critic.predict(st)

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = self.env.step(action)

            y = np.eye(self.action_size)[action:action + 1]

            xs.append(state)
            ys.append(y)
            reward_list.append(reward)
            value_list.append(val)

            score += reward
            state = next_state

            if score > 400:
                done = True

        if self.is_training:
            xs = np.vstack(xs)
            ys = np.vstack(ys)
            value_list = np.vstack(value_list)

            self.train_model(xs, ys, reward_list, value_list)

        # 에피소드마다 학습 결과 출력
        RunnerThread.recently_avg_list.append(score)
        score_avg = np.mean(RunnerThread.recently_avg_list)
        RunnerThread.all_avg_list.append(score_avg)

        # 러닝 시간
        duration = time.time() - start_time
        sec_per_step = float(duration + 1e-6)
        per = float(len(reward_list)) / sec_per_step

        print('Thread-{:d}'.format(self.id), "episode:", local_episode, "  score:{:.2f}".format(score),
              "  avg:{:.2f}".format(score_avg), '({:.2f} frame/sec)'.format(per))

    def train_model(self, states, actions, dis_rewards, values):

        dis_r = discount_rewards(dis_rewards, 0.99)
        dis_r -= dis_r.mean()
        dis_r /= dis_r.std()

        advantage = dis_r - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-8

        self.global_network.optimizer[0]([states, actions, dis_r])
        self.global_network.optimizer[1]([states, dis_r])


def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


def train():
    if not os.path.exists("save_graph"):
        os.makedirs("save_graph")

    if not os.path.exists("save_model"):
        os.makedirs("save_model")

    state_size = 60
    action_size = 3

    # Global 신경망 생성
    global_network = Agent(state_size, action_size)

    global_network.actor.summary()
    global_network.critic.summary()

    # Agent 생성
    thread_list = []

    for i in range(args.n_threads):
        tmp_agent = RunnerThread(i, state_size, action_size, global_network)
        thread_list.append(tmp_agent)

    for t in thread_list:
        t.start()

        print('Thread-{:d} start...'.format(t.id))
        time.sleep(1)

    print("Ctrl + C to close")

    save_idx = 0
    while not RunnerThread.stop_flag:
        avg_score = RunnerThread.recently_avg_list
        if len(avg_score) > 2:
            avg_score = np.mean(avg_score)

            if avg_score > 300:
                RunnerThread.stop_flag = True

        # 일정 횟수 마다 그래프 출력
        current_idx = int(RunnerThread.global_episode / 100)
        if current_idx > save_idx:
            save_idx = current_idx
            pylab.plot(RunnerThread.all_avg_list, 'b')
            pylab.savefig("./save_graph/mini_pacman_a3c_lstm.png")

        time.sleep(1)

    global_network.actor.save_weights("./save_model/minipacman_a3c_actor.h5")
    global_network.critic.save_weights("./save_model/minipacman_a3c_critic.h5")

    pylab.plot(RunnerThread.all_avg_list, 'b')
    pylab.savefig("./save_graph/mini_pacman_a3c_lstm.png")
    pylab.show()


def test():
    if not os.path.exists("save_model"):
        os.makedirs("save_model")

    state_size = 60
    action_size = 3

    # Global 신경망 생성
    global_network = Agent(state_size, action_size)

    # Agent 생성
    single_agent = RunnerThread(0, state_size, action_size, global_network, is_training=False)

    global_network.actor.load_weights("./save_model/minipacman_a3c_actor.h5")
    global_network.critic.load_weights("./save_model/minipacman_a3c_critic.h5")

    single_agent.play_episode()


if __name__ == "__main__":
    if args.mode == 'train':
        train()
    else:
        test()


