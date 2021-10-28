import gym
import numpy as np
import tensorflow as tf
import os
import argparse
import time
from gridworld import gameEnv
import cv2

parser = argparse.ArgumentParser(description="Simple 'argparse' demo application")
parser.add_argument('--mode', default='train', help='Execute mode')
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--logdir', default='./log/7.1_a3c_gridworld_distribute/4')
parser.add_argument('--max_steps', default=100001, type=int)
parser.add_argument('--env_size', default=5, type=int)
parser.add_argument('--max_ep_steps', default=50, type=int)
parser.add_argument('--worker_hosts_num', default=20, type=int)
parser.add_argument('--job_name', type=str, help="One of 'actor', 'learner'")
parser.add_argument('--task_index', type=int, default=0, help="Index of task within the job")

args = parser.parse_args()

# GPU not use
if args.job_name == 'actor':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.logging.set_verbosity(tf.logging.INFO)

# env 환경 파라메터
input_size = 42 * 42 * 3
input_shape = [42, 42, 3]
action_count = 4

buffer_size = 10000

reward_list_size = 100

def cluster_spec(num_workers, num_ps):
    """
    Tensorflow 분산 환경 설정
    """
    cluster = {}

    # Parameter server
    all_learner = []

    port = 49000
    host = 'localhost'
    for _ in range(num_ps):
        all_learner.append('{}:{}'.format(host, port))
        port += 1
    cluster['learner'] = all_learner

    # Worker
    all_actor = []

    # PC1
    port = 49100
    host = 'localhost'
    for _ in range(num_workers):
        all_actor.append('{}:{}'.format(host, port))
        port += 1

    cluster['actor'] = all_actor
    return cluster


# 보상 배열을 감쇠 적용
def calc_discount_rewards(r, gamma=0.99):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0:
            # reset the sum, since this was a game boundary (pong specific!)
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


# 신경망 구현
class A3CNetwork(object):
    def __init__(self, name, input_shape, output_size, learning_rate):
        self.name = name
        self.input_shape = input_shape
        self.output_size = output_size
        self.learning_rate = learning_rate

        with tf.variable_scope(self.name):

            # 입력값을 받는 부분
            self.states = tf.placeholder(shape=[None, *self.input_shape], dtype=tf.float32, name='input_state')
            self.actions = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32, name='input_action')
            self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
            self.rewards = tf.placeholder(tf.float32, [None], name="reward")

            # 콘볼루션을 통해 이미지 인코딩
            with tf.variable_scope('Conv'):
                net = tf.layers.conv2d(inputs=self.states, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='same',
                                       activation=tf.nn.elu)  # 11, 11
                net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[4, 4], strides=[2, 2], padding='same',
                                       activation=tf.nn.elu)  # 6, 6
                net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=[3, 3], strides=[2, 2], padding='same',
                                       activation=None)  # 3, 3

            # Normalization
            net = tf.layers.batch_normalization(inputs=net)
            net = tf.nn.elu(net)

            # Inception block
            # input : [3, 3, 128]
            # output : [3, 3, 256]
            with tf.variable_scope('inception1'):
                branch_0 = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[1, 1], activation=tf.nn.elu)

                branch_1 = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[1, 1], activation=tf.nn.elu)
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=64, kernel_size=[3, 3], activation=tf.nn.elu,
                                            padding='same')

                branch_2 = tf.layers.conv2d(inputs=net, filters=16, kernel_size=[1, 1], activation=tf.nn.elu)
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=32, kernel_size=[3, 1], activation=tf.nn.elu,
                                            padding='same')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=64, kernel_size=[1, 3], activation=tf.nn.elu,
                                            padding='same')

                branch_3 = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=1, padding='same')
                branch_3 = tf.layers.conv2d(inputs=branch_3, filters=64, kernel_size=[1, 1], activation=tf.nn.elu)

                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            # # Inception block
            # # output : [3, 3, 416]
            # with tf.variable_scope('inception2'):
            #     branch_0 = tf.layers.conv2d(inputs=net, filters=96, kernel_size=[1, 1], activation=tf.nn.elu)
            #
            #     branch_1 = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[1, 1], activation=tf.nn.elu)
            #     branch_1 = tf.layers.conv2d(inputs=branch_1, filters=128, kernel_size=[3, 3], activation=tf.nn.elu,
            #                                 padding='same')
            #
            #     branch_2 = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[1, 1], activation=tf.nn.elu)
            #     branch_2 = tf.layers.conv2d(inputs=branch_2, filters=64, kernel_size=[3, 1], activation=tf.nn.elu,
            #                                 padding='same')
            #     branch_2 = tf.layers.conv2d(inputs=branch_2, filters=128, kernel_size=[1, 3], activation=tf.nn.elu,
            #                                 padding='same')
            #
            #     branch_3 = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=1, padding='same')
            #     branch_3 = tf.layers.conv2d(inputs=branch_3, filters=64, kernel_size=[1, 1], activation=tf.nn.elu)
            #
            #     net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=1)

            net = tf.layers.flatten(net)
            net = tf.layers.dense(inputs=net, units=256)

            # Normalization
            net = tf.layers.batch_normalization(inputs=net)
            net = tf.nn.elu(net)

            # LSTM
            _rnn_out_size = 128
            _rnn_in = tf.expand_dims(net, [0])
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=_rnn_out_size, state_is_tuple=True)
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
            step_size = tf.shape(self.states)[:1]
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                inputs=_rnn_in, cell=lstm_cell, initial_state=state_in, sequence_length=step_size,
                time_major=False, dtype=tf.float32)
            lstm_c, lstm_h = lstm_state

            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])

            # 최종을 다시 벡터화
            net = tf.reshape(lstm_outputs, [-1, _rnn_out_size])

            # Normalization
            net = tf.layers.batch_normalization(inputs=net)

            # Output
            self.pred = tf.layers.dense(inputs=net, units=self.output_size, activation=tf.nn.softmax, name='pred')
            self.values = tf.squeeze(tf.layers.dense(inputs=net, units=1, name="values"))

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Agent():
    max_ep_steps = 50  # episode별 최대 걸음 수

    def __init__(self, id, network, input_shape, action_count):
        super(Agent, self).__init__()

        self.id = id
        self.network = network
        self.input_shape = input_shape
        self.action_count = action_count

        self.env = gameEnv(partial=False, size=args.env_size)

    # State 전처리
    def preProcessState(self, states):
        resize_img = cv2.resize(states, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        return resize_img

    def play_episode(self, sess):
        # Set episode start time
        start_time = time.time()

        s = self.env.reset()
        s = self.preProcessState(s)

        done = False
        episode_reward = 0
        episode_step = 0

        # 에피스드 임시 버퍼
        episode_buffer = []

        rnn_state = self.network.state_init

        while not done:
            episode_step += 1

            # Episode별 XX회 이상되면 종료
            if episode_step > Agent.max_ep_steps:
                break

            # Action 추출
            feed = {
                self.network.states: [s],
                self.network.state_in[0]: rnn_state[0],
                self.network.state_in[1]: rnn_state[1]
            }
            action_prob, v, rnn_state = sess.run([self.network.pred, self.network.values, self.network.state_out],
                                                 feed_dict=feed)
            action_prob = np.squeeze(action_prob)
            a = np.random.choice(self.action_count, size=1, p=action_prob)[0]

            # 주어진 행동을 실행하고 다음 상태, 보상, 종료 여부를 가져옴
            s1, r, done = self.env.step(a)

            # Action에 one_hot 적용
            action_one_hot = np.squeeze(np.eye(self.action_count)[a:a + 1])

            # 버퍼에 현재 상태, 행동, 보상, 다음 상태, 종료 여부를 저장한다
            episode_buffer.append([s, action_one_hot, r, v])

            # State 변경
            s1 = self.preProcessState(s1)
            s = s1

            episode_reward += r

        # discound_rewards & advantage
        episode_buffer = np.reshape(episode_buffer, [-1, 4])

        temp_r = episode_buffer[:, 2]
        discount_rewards = calc_discount_rewards(temp_r)
        discount_rewards -= np.mean(discount_rewards)
        discount_rewards /= np.std(discount_rewards) + 1e-8

        temp_v = episode_buffer[:, 3]
        advantage = discount_rewards - temp_v
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-8

        temp_buffer = []

        for idx in range(len(episode_buffer)):
            bf = episode_buffer[idx]

            xs = np.ravel(bf[0])
            ys = np.ravel(bf[1])
            r = bf[2]
            v = bf[3]

            dis_r = discount_rewards[idx]
            adv = advantage[idx]

            temp_buffer.append(np.hstack([xs, ys, r, v, dis_r, adv]))

        episode_buffer = temp_buffer

        # 러닝 시간
        duration = time.time() - start_time
        frame_sec = episode_step / float(duration + 1e-6)

        return episode_buffer, episode_reward, frame_sec

# Model 학습
def train(sess, network, buffer, op):

    xs = np.vstack(buffer[:, :input_size])
    xs = np.reshape(xs, [-1, *input_shape])
    ys = np.vstack(buffer[:, input_size:input_size + 4])

    rnn_state = network.state_init

    discount_rewards = buffer[:, input_size + 6]
    advantage = buffer[:, input_size + 7]

    feed = {
        network.states: xs,
        network.actions: ys,
        network.rewards: discount_rewards,
        network.advantages: advantage,
        network.state_in[0]: rnn_state[0],
        network.state_in[1]: rnn_state[1]
    }

    _ = sess.run(op, feed)


# 학습용 method
def main_train():

    # Network
    spec = cluster_spec(args.worker_hosts_num, 1)
    cluster = tf.train.ClusterSpec(spec)

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=args.job_name,
                             task_index=args.task_index)

    local_job_device = "/job:%s/task:%d" % (args.job_name, args.task_index)
    shared_job_device = '/job:learner/task:0'

    is_learner = (args.job_name == 'learner')

    filters = [shared_job_device, local_job_device]
    sess_config = tf.ConfigProto(device_filters=filters, allow_soft_placement=True)

    # Global network
    with tf.device(shared_job_device + '/cpu'):
        global_network = A3CNetwork('global', input_shape, action_count, args.learning_rate)

        # Global step
        global_step = tf.train.get_or_create_global_step()

        # Reward list
        reward_list = tf.Variable(tf.zeros([reward_list_size]), trainable=False, dtype=tf.float32,
                                  name='reward_list')
        reward_in = tf.placeholder(dtype=tf.float32, shape=[reward_list_size], name='reward')
        op_reward_add = reward_list.assign(reward_in)

        buffer_experience = tf.FIFOQueue(buffer_size, dtypes=[tf.float32], shapes=[input_size + action_count + 4], shared_name='buffer_experience')
        buffer_step_length = tf.FIFOQueue(5, dtypes=tf.int32, shapes=[1], shared_name='buffer_step_length')
        buffer_reward = tf.FIFOQueue(5, dtypes=tf.float32, shapes=[1], shared_name='buffer_reward')

        step_length = buffer_step_length.dequeue()

        op_deque_bf = buffer_experience.dequeue_many(step_length)
        op_deque_reward = buffer_reward.dequeue()

        get_bf_experience_size = buffer_experience.size()

    if is_learner == True:
        print('Learner server starting...')

        with tf.device(shared_job_device):
            # Loss 계산
            _policy_gain = -tf.reduce_sum(tf.log(global_network.pred + 1e-5) * global_network.actions,
                                          axis=1) * global_network.advantages
            _policy_gain = tf.reduce_mean(_policy_gain)
            _entropy = - tf.reduce_sum(global_network.pred * tf.log(global_network.pred + 1e-5), axis=1)
            _entropy = tf.reduce_mean(_entropy)
            _value_loss = tf.losses.mean_squared_error(global_network.values, global_network.rewards, scope="value_loss")

            total_loss = _policy_gain + (_value_loss * 0.5) - (_entropy * 0.01)

            opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            # train_op = opt.minimize(total_loss, global_step=global_step)

            gradients = opt.compute_gradients(total_loss, global_network.var_list)
            cliped_grads = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in gradients]

            apply_gradients = opt.apply_gradients(cliped_grads, global_step=global_step)

        scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer())

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(args.task_index == 0),
                                               hooks=[],
                                               scaffold=scaffold,
                                               config=sess_config) as sess:
            # 모델 그래프 최종 확정
            # tf.get_default_graph().finalize()

            print('Model training starting...')

            while not sess.should_stop():
                # Set episode start time
                start_time = time.time()

                bf_flag = sess.run(get_bf_experience_size)

                if bf_flag == 0:
                    print('Buffer empty...', time.time())
                    time.sleep(1)
                    continue

                buffer, ep_r = sess.run([op_deque_bf, op_deque_reward])

                train(sess, global_network, buffer, apply_gradients)

                r_list = sess.run(reward_list)
                r_list = np.append(r_list, ep_r, axis=0)
                r_list = r_list[1:]
                sess.run(op_reward_add, feed_dict={reward_in: r_list})

                avg_reward = np.mean(r_list)

                # 러닝 시간
                duration = float(time.time() - start_time)
                step_sec = 1 / duration

                print('Step :', sess.run(global_step),
                      'avg_reward : {:.2f}'.format(avg_reward),
                      '{:.1f} step/sec'.format(step_sec))

    else:
        print('Actor server starting...')

        with tf.device(local_job_device + '/cpu'):
            local_network = A3CNetwork('local', input_shape, action_count, args.learning_rate)

            ph_ep_buffer = tf.placeholder(dtype=tf.float32, shape=[None, input_size + action_count + 4])
            ph_ep_step_length = tf.placeholder(dtype=tf.int32, shape=[1])
            ph_ep_reward = tf.placeholder(dtype=tf.float32, shape=[1])

            op_add_ep_buffer = buffer_experience.enqueue_many(ph_ep_buffer)
            op_add_step_length = buffer_step_length.enqueue(ph_ep_step_length)
            op_add_ep_reward = buffer_reward.enqueue(ph_ep_reward)


        # Weight sync global to local
        sync_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(local_network.var_list, global_network.var_list)])

        agent = Agent(args.task_index, local_network, input_shape, action_count)

        is_chief = (args.task_index == 0)

        # Initializer
        local_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='local')
        local_init_op = tf.variables_initializer(local_variables)

        scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(),
                                     ready_for_local_init_op=local_init_op)

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               hooks=[],
                                               scaffold=scaffold,
                                               config=sess_config) as sess:

            # # 모델 그래프 최종 확정
            # tf.get_default_graph().finalize()

            print('Actor starting...')

            while not sess.should_stop():

                bf_size = sess.run(get_bf_experience_size)
                if bf_size > (buffer_size * 0.9):
                    message = "Task: {}, Buffer size : {} , wait...)".format(args.task_index, bf_size)
                    print(message)
                    time.sleep(1)
                    continue

                sess.run(sync_op)

                ep_b, ep_r, frame_sec = agent.play_episode(sess)

                sess.run([op_add_ep_buffer, op_add_step_length, op_add_ep_reward],
                         feed_dict={ph_ep_buffer: ep_b,
                                    ph_ep_step_length: [len(ep_b)],
                                    ph_ep_reward: [ep_r]})

                message = "Task: {}, reward= {:.2f}, ({:.2f} frame/sec), ep_stpes= {}".format(
                    args.task_index, ep_r, frame_sec, len(ep_b))
                print(message)


# Test용 method
def main_test():
    pass

if __name__ == '__main__':
    if args.mode == 'train':
        main_train()
    else:
        main_test()
