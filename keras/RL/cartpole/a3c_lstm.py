import os
import gym
import pylab
import numpy as np
from keras.layers import Dense, Input, LSTM, Reshape, TimeDistributed
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import threading
import argparse
import time
from collections import deque

np.random.seed(777)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description="Simple 'argparse' demo application")
parser.add_argument('--mode', default='train', help='Execute mode')
parser.add_argument('--learning_rate', default=1e-2, type=float)
parser.add_argument('--logdir', default='./log/a3c_lstm_log')
parser.add_argument('--max_steps', default=1000, type=int)
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
        input = Input(shape=[1, self.state_size, ])
        # fc = TimeDistributed(Dense(16, activation='relu'))(input)

        net = LSTM(16, activation='relu', return_sequences=True)(input)
        net = LSTM(16, activation='relu')(net)

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
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        # 탐색을 지속적으로 하기 위한 엔트로피 오류
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        # 두 오류함수를 더해 최종 오류함수를 만듬
        loss = cross_entropy + 0.01 * entropy

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
    score_list = deque(maxlen=30)
    score_list2 = []

    def __init__(self, id, state_size, action_size, global_network):
        super(RunnerThread, self).__init__()

        self.env = gym.make('CartPole-v1')

        self.id = id

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        self.render = False

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

        # Set episode start time
        start_time = time.time()

        self.update_local_model()

        done = False
        score = 0

        xs, ys, reward_list, value_list = [], [], [], []

        # env 초기화
        state = self.env.reset()

        while not done:
            if self.render:
                self.env.render()

            # 현재 상태로 행동을 선택
            st = np.reshape(state, [1, 1, self.state_size])

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

            if done:
                xs = np.vstack(xs)
                xs = np.reshape(xs, (len(xs), 1, self.state_size))
                ys = np.vstack(ys)
                value_list = np.vstack(value_list)

                self.train_model(xs, ys, reward_list, value_list)

                # 에피소드마다 학습 결과 출력
                RunnerThread.score_list.append(score)
                RunnerThread.score_list2.append(score)

                # 러닝 시간
                duration = time.time() - start_time
                sec_per_step = float(duration)
                per = float(len(reward_list)) / sec_per_step

                print('Thread-{:d}'.format(self.id), "episode:", local_episode, "  score:", score,
                      '({:.2f} frame/sec)'.format(per))

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


def main():
    if not os.path.exists("save_graph"):
        os.makedirs("save_graph")

    if not os.path.exists("save_model"):
        os.makedirs("save_model")

    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    env.close()

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

    while not RunnerThread.stop_flag:
        avg_score = RunnerThread.score_list
        if len(avg_score) > 2:
            avg_score = np.mean(avg_score)

            if avg_score > 490:
                RunnerThread.stop_flag = True

        time.sleep(1)

    pylab.plot(RunnerThread.score_list2, 'b')
    pylab.savefig("./save_graph/cartpole_a3c_lstm.png")
    pylab.show()


if __name__ == "__main__":
    main()

