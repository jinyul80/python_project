import sys
import gym
import pylab
import numpy as np
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

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
        self.learning_rate = 1e-2

        # 모델과 옵티마이저 설정
        self.model = self.build_model()
        self.optimizer = self.optimizer()

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_pg_trained.h5")

    # 상태가 입력, 행동이 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))
        model.add(Activation('softmax'))
        model.summary()

        return model

    # 정책신경망을 업데이트 하기 위한 오류함수와 훈련함수의 생성
    def optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        discounted_rewards = K.placeholder(shape=[None, ])

        # 크로스 엔트로피 오류함수 계산
        action_prob = K.sum(action * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)

        # 정책신경망을 업데이트하는 훈련함수 생성
        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [],
                                        loss)
        train = K.function([self.model.input, action, discounted_rewards], [],
                           updates=updates)

        return train

    # 정책으로 행동 선택
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        action = self.model.predict(state)[0]
        return np.random.choice(self.action_size, size=1, p=action)[0]

    # 모델 학습
    def train_model(self, states, actions, dis_rewards):
        self.optimizer([states, actions, dis_rewards])


def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # DQN 에이전트 생성
    agent = Agent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0

        xs, ys, reward_list = [], [], []

        # env 초기화
        state = env.reset()

        while not done:
            if agent.render:
                env.render()

            # 현재 상태로 행동을 선택
            action = agent.get_action(state)

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(action)

            y = np.eye(action_size)[action:action+1]

            xs.append(state)
            ys.append(y)
            reward_list.append(reward)

            score += reward
            state = next_state

            if done:

                dis_rewards = discount_rewards(reward_list, 0.99)
                dis_rewards -= dis_rewards.mean()
                dis_rewards /= dis_rewards.std()

                xs = np.vstack(xs)
                ys = np.vstack(ys)

                agent.train_model(xs, ys, dis_rewards)

                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_pg.png")
                print("episode:", e, "  score:", score)

                # 이전 20개 에피소드의 점수 평균이 490보다 크면 학습 중단
                if np.mean(scores[-min(20, len(scores)):]) > 490:
                    agent.model.save_weights("./save_model/cartpole_pg.h5")
                    sys.exit()