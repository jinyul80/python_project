import os
import time
import sys
import gym
import pylab
import numpy as np
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K

# GPU 사용 않함
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

EPISODES = 500

np.random.seed(777)

# 카트폴 예제에서의 에이전트
class Agent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # 하이퍼파라미터
        self.actor_lr = 1e-2
        self.critic_lr = 1e-2

        # 정책신경망과 가치신경망 생성
        self.actor, self.critic = self.build_model()
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

    # 신경망 생성
    def build_model(self):
        input = Input(shape=[self.state_size, ])
        fc = Dense(24, activation='relu')(input)

        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        # 가치와 정책을 예측하는 함수를 만들어냄
        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

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

    # 모델 학습
    def train_model(self, states, actions, dis_rewards, values):

        dis_r = discount_rewards(dis_rewards, 0.99)
        dis_r -= dis_r.mean()
        dis_r /= dis_r.std()

        advantage = dis_r - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-8

        self.optimizer[0]([states, actions, dis_r])
        self.optimizer[1]([states, dis_r])



def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


if __name__ == "__main__":
    if not os.path.exists("save_graph"):
        os.makedirs("save_graph")

    if not os.path.exists("save_model"):
        os.makedirs("save_model")

    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # DQN 에이전트 생성
    agent = Agent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):

        # Set episode start time
        start_time = time.time()

        done = False
        score = 0

        xs, ys, reward_list, value_list = [], [], [], []

        # env 초기화
        state = env.reset()

        while not done:
            if agent.render:
                env.render()

            # 현재 상태로 행동을 선택
            st = np.reshape(state, [1, state_size])
            action = agent.get_action(st)
            val = agent.critic.predict(st)

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(action)

            y = np.eye(action_size)[action:action+1]

            xs.append(state)
            ys.append(y)
            reward_list.append(reward)
            value_list.append(val)

            score += reward
            state = next_state

            if done:

                xs = np.vstack(xs)
                ys = np.vstack(ys)
                value_list = np.vstack(value_list)

                agent.train_model(xs, ys, reward_list, value_list)

                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_a2c.png")

                # 러닝 시간
                duration = time.time() - start_time
                sec_per_step = float(duration)
                per = float(len(reward_list)) / sec_per_step

                print("episode:", e, "  score:", score, '({:.2f} frame/sec)'.format(per))

                # 이전 30개 에피소드의 점수 평균이 490보다 크면 학습 중단
                if np.mean(scores[-min(30, len(scores)):]) > 490:
                    agent.actor.save_weights("./save_model/cartpole_a2c_actor.h5")
                    agent.critic.save_weights("./save_model/cartpole_a2c_critic.h5")

                    pylab.show()

                    sys.exit()