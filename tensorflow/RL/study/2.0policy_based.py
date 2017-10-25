import numpy as np
import _pickle as pickle # python 3.5
import tensorflow as tf
import matplotlib.pyplot as plt
import math

import gym
env = gym.make('CartPole-v0')

# 환경을 초기화한다. 초기 상태(state, observation을 만드는 env.reset())
env.reset()

# 하이퍼파라미터
H = 24 # 은닉층의 노드 수
batch_size = 5 # 몇개의 에피소드마다 파라미터를 업데이트할 것인지
learning_rate = 1e-2 # 학습률
gamma = 0.99 # 보상에 대한 할인 인자

D = 4 # 입력 차원

# 그래프를 초기화한다
tf.reset_default_graph()

# 관찰은 상태를 받는다.
observations = tf.placeholder(tf.float32, [None, D], name="input_x")

# tf.layers 사용
W1 = tf.layers.dense(inputs=observations, units=H, activation=tf.nn.relu, use_bias=False)
probability = tf.layers.dense(inputs=W1, units=1, activation=tf.nn.sigmoid, use_bias=False)

# W1은 은닉층으로 보낸다
# W1 = tf.get_variable("W1", shape=[D, H],
#            initializer=tf.contrib.layers.xavier_initializer())
# relu 활성화함수를 쓴다
# layer1 = tf.nn.relu(tf.matmul(observations, W1))
# 은닉층의 결과인 10개의 값으로 하나의 결과값(점수)을 낸다
# W2 = tf.get_variable("W2", shape=[H, 1],
#            initializer=tf.contrib.layers.xavier_initializer())
# layer2 = tf.matmul(layer1, W2)
# 점수를 확률로 변환한다.
# probability = tf.nn.sigmoid(layer2)

# 학습 가능한 변수들 (가중치)
tvars = tf.trainable_variables()
# 출력값을 받는 부분
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
# 이득을 받는 부분
advantages = tf.placeholder(tf.float32, name="reward_signal")

# 손실함수. 좋은 이득(시간 경과에 따른 보상)을 더 자주 주는 행동으로
# 가중치를 보내고, 덜 가능성이 있는 행동에 가중치를 보낸다.

# 왜 이게 동작하는 것일까?
# cross entropy와 비슷하다. 내가 행동을 1로 했고, 그 행동에 높은 확률을 주었다면 손실이 작고,
# 내가 0으로 움직였고, 그 행동에 낮은 확률을 주었다면 손실이 작다.
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
# 위의 각 행동의 잘하고 못하고 부분을 지연된 보상으로 조정하고 난 모든 것을 손실로 본다.
loss = -tf.reduce_mean(loglik * advantages)
# 이 손실을 이용해 학습 변수들의 그라디언트를 구한다.
newGrads = tf.gradients(loss, tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
# 여러 에피소드로부터의 그라디언트를 모았다가 그것을 적용한다.
# 히든 layer의 개수 만큼 그라디언트 필요
# 왜 매 에피소드마다 그라디언트를 업데이트하지 않느냐면 에피소드의 노이즈까지 학습할까봐
adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # 최적화기 adam
W1Grad = tf.placeholder(tf.float32, name="batch_grad1") # 그라디언트 저장하는 부분
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
# 그라디언트 적용하는 부분
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))


# 할인 함수
def discount_rewards(r):
    """ 보상 배열을 받아 할인된 보상을 계산한다"""
    # 할인된 보상을 전부 0으로 초기화
    discounted_r = np.zeros_like(r)
    # 보상을 역순으로 더해가며, 가중치를 준다.
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# 입력값들, 출력값들, 보상들을 저장하는 리스트
xs, drs, ys = [], [], []
# 보상은 없다고 생각
EPISODE_100_REWARD_LIST = []
reward_sum = 0
# 에피소드 수를 기록
episode_number = 1
# 에피소드 몇번 할지 기록
total_episodes = 10000
# 텐서플로 변수를 초기화함
init = tf.global_variables_initializer()

# 텐서플로 실행
sess = tf.InteractiveSession()
# 에이전트 표시 안함
rendering = False
# 변수 초기화
sess.run(init)
# 상태 초기화
observation = env.reset()

# 그라디언트 담을 곳 초기화
# 정책 신경망을 업데이트 하기 전까지 그라디언트를 모은다.
gradBuffer = sess.run(tvars)
# 전부 0으로 초기화
# for ix, grad in enumerate(gradBuffer):
#     gradBuffer[ix] = grad * 0

# 에피소드 시작
while episode_number <= total_episodes:

    # Rendering the environment slows things down,
    # 에이전트를 표시하는 것은 학습을 느리게 한다
    # 그래서 에이전트가 잘 작동할 때까지 표시 안한다

    if reward_sum / batch_size > 100 or rendering == True:
        # env.render()
        rendering = True

    # 상태를 신경망이 다룰 수 있는 형태로 바꿈
    x = np.reshape(observation, [1, D])

    # 정책 신경망을 돌려서 액션에 대한 확률 값을 얻은
    tfprob = sess.run(probability, feed_dict={observations: x})
    # 확률 값보다 무작위 값이 작다면 1로 움직이고
    # 그렇지 않다면 0으로 움직임
    action = 1 if np.random.uniform() < tfprob else 0

    xs.append(x)  # 상태를 저장한다
    y = 1 if action == 0 else 0  # 가짜 라벨, 각 행동에 대한 라벨을 저장함
    ys.append(y)

    # 새로운 상태와 보상을 얻음
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    # 보상을 기록함
    # step 함수를 부르고나면 이전 행동에 대한 보상을 잃기 때문에 기록함
    drs.append(reward)

    # 에피소드가 끝나면
    if done:
        episode_number += 1

        # 에피소드별 보상 이력 추가
        EPISODE_100_REWARD_LIST.append(reward_sum)
        if len(EPISODE_100_REWARD_LIST) > 100:
            EPISODE_100_REWARD_LIST = EPISODE_100_REWARD_LIST[1:]

        # 보상을 초기화
        reward_sum = 0

        # 에피소드별 각 상태, 라벨, 보상으로 업데이트를 준비함
        epx = np.vstack(xs)
        epy = np.vstack(ys)
        epr = np.vstack(drs)
        xs, drs, ys = [], [], []  # 다음 에피소드를 위해 초기화

        # 시간에 대해 보상들을 할인함
        discounted_epr = discount_rewards(epr)
        # 보상들을 평균이 0이고 분산이 1이 되도록 정규화
        # 그라디언트의 분산을 조절하는데 도움을 준다.
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # 이 에피소드의 그라디언트를 구함
        tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})

        # 그라디언트를 그라디언트 버퍼에 저장함
        for ix, grad in enumerate(tGrad):
            gradBuffer[ix] += grad

        # 충분한 에피소드(배치 사이즈)만큼이 끝나면,
        # 그라디언트 버퍼에 저장된 그라디언트를 신경망에 적용함
        if episode_number % batch_size == 0:
            sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
            for ix, grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0

            # 얼마나 우리 신경망이 에피소드 별로 잘하는지 통계를 냄
            print('Episode : %d, Average reward for last 5 episode %.2f, Average reward for last 100 episode %.2f.' % (
                episode_number, np.mean(EPISODE_100_REWARD_LIST[-5:]), np.mean(EPISODE_100_REWARD_LIST)))

            # 에피소드 100회 평균 보상이 195을 넘으면 멈춤
            if np.mean(EPISODE_100_REWARD_LIST) >= 195:
                print("Task solved in", episode_number, 'episodes!')
                break

        # 상태를 초기화
        observation = env.reset()

env.close()
sess.close()

print(episode_number, 'Episodes completed.')



