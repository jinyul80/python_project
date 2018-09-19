import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import argparse
import time

import mini_pacman

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description="Simple 'argparse' demo application")
parser.add_argument('--mode', default='train', help='Execute mode')
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--logdir', default='./log/7.1_a3c_pacman_log/new')
parser.add_argument('--max_steps', default=5001, type=int)
parser.add_argument('--worker_hosts_num', default=10, type=int)
parser.add_argument('--job_name', type=str, help="One of 'ps', 'worker'")
parser.add_argument('--task_index', type=int, default=0, help="Index of task within the job")

args = parser.parse_args()

# GPU not use
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.logging.set_verbosity(tf.logging.ERROR)

def cluster_spec(num_workers, num_ps):
    """
    Tensorflow 분산 환경 설정
    """
    cluster = {}

    # Parameter server
    all_ps = []

    port = 49000
    host = '192.168.1.253'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    # Worker
    all_workers = []

    # PC1
    port = 49100
    host = '192.168.1.253'
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1

    # PC2
    port = 49100
    host = '192.168.1.252'
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1

    # PC3
    port = 49100
    host = '192.168.1.251'
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1

    cluster['worker'] = all_workers
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
    def __init__(self, name, input_size, input_shape, output_size, learning_rate):
        self.name = name
        self.input_size = input_size
        self.input_shape = input_shape
        self.output_size = output_size
        self.learning_rate = learning_rate

        self._build()

    def _build(self):
        with tf.variable_scope(self.name):
            # 입력값을 받는 부분 21168 차원은 84*84*3 의 차원이다.
            self.states = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32, name='input_state')
            self.actions = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32, name='input_action')
            self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
            self.rewards = tf.placeholder(tf.float32, [None], name="reward")

            self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='prev_rewards')
            self.prev_actions = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32, name='prev_actions')

            _imageIn = tf.reshape(self.states, shape=[-1, *self.input_shape])

            # Dense 레이어
            net = tf.layers.dense(inputs=_imageIn, units=256, activation=tf.nn.relu)
            net = tf.layers.dense(inputs=net, units=128)

            # Normalization
            net = tf.layers.batch_normalization(inputs=net)
            net = tf.nn.relu(net)

            # Concat Prev rewards, actions
            net = tf.concat(axis=1, values=[net, self.prev_rewards, self.prev_actions])

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
            step_size = tf.shape(_imageIn)[:1]
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

            # Loss 계산
            _policy_gain = -tf.reduce_sum(tf.log(self.pred + 1e-5) * self.actions, axis=1) * self.advantages
            _policy_gain = tf.reduce_mean(_policy_gain)
            _entropy = - tf.reduce_sum(self.pred * tf.log(self.pred + 1e-5), axis=1)
            _entropy = tf.reduce_mean(_entropy)
            _value_loss = tf.losses.mean_squared_error(self.values, self.rewards, scope="value_loss")

            self.total_loss = _policy_gain + (_value_loss * 0.5) - (_entropy * 0.01)

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Agent():

    def __init__(self, id, input_size, input_shape, output_dim, learning_rate, cluster, is_training=True):
        super(Agent, self).__init__()

        self.id = id

        self.input_size = input_size
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = mini_pacman.Gym(show_game=False)

        self.is_training = is_training

        self.reward_list_size = 100

        worker_device = "/job:worker/task:%d" % id
        replica_device = tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % id,
            ps_device="/job:ps",
            cluster=cluster)

        # Global network
        with tf.device(replica_device):
            self.global_network = A3CNetwork('global', input_size, input_shape, output_dim, learning_rate)

            self.global_step = tf.train.get_or_create_global_step()
            self.global_step_add = self.global_step.assign_add(1)

            self.reward_list = tf.Variable(tf.zeros([self.reward_list_size]), trainable=False, dtype=tf.float32,
                                           name='reward_list')
            self.reward_in = tf.placeholder(dtype=tf.float32, shape=[self.reward_list_size], name='reward')
            self.reward_add_op = self.reward_list.assign(self.reward_in)

        # Local network
        with tf.device(worker_device):
            self.local = A3CNetwork('local', input_size, input_shape, output_dim, learning_rate)

            grads = tf.gradients(self.local.total_loss, self.local.var_list)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)

        # Train op
        grads_and_vars = list(zip(grads, self.global_network.var_list))

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = opt.apply_gradients(grads_and_vars)

        # Weight sync global to local
        self.sync_op = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(self.local.var_list, self.global_network.var_list)])

    # State 전처리
    def preProcessState(self, states):
        return np.reshape(states, [self.input_size])

    def play_episode(self, sess):
        # Set episode start time
        start_time = time.time()

        step = sess.run(self.global_step)
        sess.run(self.global_step_add)

        # Variable 동기화
        sess.run(self.sync_op)

        s = self.env.reset()
        s = self.preProcessState(s)

        done = False
        episode_reward = 0
        episode_step = 0

        # 에피스드 임시 버퍼
        episode_buffer = []

        rnn_state = self.local.state_init

        r = 0
        action_one_hot = np.zeros(self.output_dim)

        while not done:
            episode_step += 1

            # Action 추출
            feed = {
                self.local.states: [s],
                self.local.state_in[0]: rnn_state[0],
                self.local.state_in[1]: rnn_state[1],
                self.local.prev_rewards: [[r]],
                self.local.prev_actions: [action_one_hot]
            }
            action_prob, v, rnn_state = sess.run([self.local.pred, self.local.values, self.local.state_out],
                                                 feed_dict=feed)
            action_prob = np.squeeze(action_prob)
            a = np.random.choice(self.output_dim, size=1, p=action_prob)[0]

            # 주어진 행동을 실행하고 다음 상태, 보상, 종료 여부를 가져옴
            s1, r, done, _ = self.env.step(a)

            # Action에 one_hot 적용
            action_one_hot = np.squeeze(np.eye(self.output_dim)[a:a + 1])

            # 버퍼에 현재 상태, 행동, 보상, 다음 상태, 종료 여부를 저장한다
            episode_buffer.append([s, action_one_hot, r, v])

            # State 변경
            s1 = self.preProcessState(s1)
            s = s1

            episode_reward += r

            if episode_reward > 400:
                break

        # Episode 보상 출력 및 global에 기록
        r_list = sess.run(self.reward_list)
        r_list = np.append(r_list, [episode_reward], axis=0)
        r_list = r_list[1:]

        avg_reward = np.mean(r_list)

        sess.run(self.reward_add_op, feed_dict={self.reward_in: r_list})

        # Traing 모드일 경우 학습
        if self.is_training:
            self.train(sess, episode_buffer)

        # 러닝 시간
        duration = time.time() - start_time
        frame_sec = episode_step / float(duration + 1e-6)

        return step, episode_reward, avg_reward, frame_sec

    # Model 학습
    def train(self, sess, buffer):

        buffer = np.reshape(buffer, [-1, 4])
        xs = np.vstack(buffer[:, 0])
        ys = np.vstack(buffer[:, 1])
        train_rewards = buffer[:, 2]
        values = buffer[:, 3]

        prev_r = np.vstack([0] + train_rewards[:-1].tolist())
        prev_a = np.vstack([np.zeros(self.output_dim), ys[:-1].tolist()])

        rnn_state = self.local.state_init

        discount_rewards = calc_discount_rewards(train_rewards)
        discount_rewards -= np.mean(discount_rewards)
        discount_rewards /= np.std(discount_rewards) + 1e-8

        advantage = discount_rewards - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-8

        feed = {
            self.local.states: xs,
            self.local.actions: ys,
            self.local.rewards: discount_rewards,
            self.local.advantages: advantage,
            self.local.state_in[0]: rnn_state[0],
            self.local.state_in[1]: rnn_state[1],
            self.local.prev_rewards: prev_r,
            self.local.prev_actions: prev_a
        }

        _ = sess.run(self.train_op, feed)


# 학습용 method
def main_train():

    # Network
    spec = cluster_spec(args.worker_hosts_num, 1)
    cluster = tf.train.ClusterSpec(spec)

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=args.job_name,
                             task_index=args.task_index)

    if args.job_name == "ps":
        print('Parameter server starting...')
        server.join()

    elif args.job_name == "worker":
        print('Worker server starting...')

        # env 환경 파라메터
        input_size = 60
        input_shape = [60]
        output_dim = 3

        single_agent = Agent(id=args.task_index,
                             input_size=input_size,
                             input_shape=input_shape,
                             output_dim=output_dim,
                             learning_rate=args.learning_rate,
                             cluster=cluster)

        is_chief = (args.task_index == 0)

        # Initializer
        local_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='local')
        local_init_op = tf.variables_initializer(local_variables)

        saver_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global')
        saver = tf.train.Saver(var_list=saver_var_list)

        sess_config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:%d" % args.task_index])

        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir=args.logdir,
                                 init_op=tf.global_variables_initializer(),
                                 local_init_op=local_init_op,
                                 saver=saver,
                                 summary_op=None,
                                 save_model_secs=300,
                                 global_step=single_agent.global_step)

        with sv.managed_session(server.target, config=sess_config) as sess:

            # 모델 그래프 최종 확정
            tf.get_default_graph().finalize()

            print('Model training starting...')

            before_step = sess.run(tf.train.get_global_step()) - 1
            before_time = time.time()

            while not sv.should_stop():
                step, ep_r, avg_r, frame_sec = single_agent.play_episode(sess)

                message = "Task: {}, Episode: {}, reward= {:.2f}, avg= {:.2f} ({:.2f} frame/sec)".format(args.task_index, step, ep_r, avg_r, frame_sec)
                print(message)

                n_step = step - before_step
                e_time = time.time()
                step_per_sec = n_step / float((e_time - before_time) + 1e-6)

                before_step = step
                before_time = e_time

                if is_chief:
                    summary = tf.Summary()
                    summary.value.add(tag='avg_reward', simple_value=float(avg_r))
                    # summary.value.add(tag='frame/sec', simple_value=float(frame_sec))
                    summary.value.add(tag='step/sec', simple_value=float(step_per_sec))
                    sv.summary_computed(sess, summary, global_step=step)

                if np.mean(avg_r) > 350:
                    print('Model training success...')
                    sv.request_stop()



# Test용 method
def main_test():
    pass

if __name__ == '__main__':
    if args.mode == 'train':
        main_train()
    else:
        main_test()
