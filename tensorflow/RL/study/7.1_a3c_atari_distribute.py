import gym
import numpy as np
import tensorflow as tf
import os
import argparse
import time
import cv2

parser = argparse.ArgumentParser(description="Simple 'argparse' demo application")
parser.add_argument('--mode', default='train', help='Execute mode')
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--logdir', default='./log/7.1_a3c_atari_log/pong-1')
parser.add_argument('--max_steps', default=1000001, type=int)
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
    def __init__(self, name, input_shape, output_size, learning_rate):
        self.name = name
        self.input_shape = input_shape
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.reward_list_size = 100

        self._build()

    def _build(self):
        with tf.variable_scope(self.name):

            # 입력값을 받는 부분
            self.states = tf.placeholder(shape=[None, *self.input_shape], dtype=tf.float32, name='input_state')
            self.actions = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32, name='input_action')
            self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
            self.rewards = tf.placeholder(tf.float32, [None], name="reward")

            self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='prev_rewards')
            self.prev_actions = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32, name='prev_actions')

            # 콘볼루션을 통해 이미지 인코딩
            with tf.variable_scope('Conv'):
                net = tf.layers.conv2d(inputs=self.states, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='same',
                                       activation=tf.nn.elu)  # 27, 20, 32
                net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[8, 8], strides=[4, 4], padding='same',
                                       activation=tf.nn.elu)  # 7, 5, 64
                net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=[3, 3], strides=[2, 2], padding='same')
                # 4, 3, 128

            # Normalization
            net = tf.layers.batch_normalization(inputs=net)
            net = tf.nn.elu(net)

            # Inception block
            # input : [4, 3, 128]
            # output : [4, 3, 256]
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

            # Inception block
            # input : [4, 3, 256]
            # output : [4, 3, 416]
            with tf.variable_scope('inception2'):
                branch_0 = tf.layers.conv2d(inputs=net, filters=96, kernel_size=[1, 1], activation=tf.nn.elu)

                branch_1 = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[1, 1], activation=tf.nn.elu)
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=128, kernel_size=[3, 3], activation=tf.nn.elu,
                                            padding='same')

                branch_2 = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[1, 1], activation=tf.nn.elu)
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=64, kernel_size=[3, 1], activation=tf.nn.elu,
                                            padding='same')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=128, kernel_size=[1, 3], activation=tf.nn.elu,
                                            padding='same')

                branch_3 = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=1, padding='same')
                branch_3 = tf.layers.conv2d(inputs=branch_3, filters=64, kernel_size=[1, 1], activation=tf.nn.elu)

                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            net = tf.layers.max_pooling2d(inputs=net, pool_size=[4, 3], strides=[1, 1])

            net = tf.layers.flatten(net)
            net = tf.layers.dense(inputs=net, units=512)

            # Normalization
            net = tf.layers.batch_normalization(inputs=net)
            net = tf.nn.elu(net)

            # Concat Prev rewards, actions
            net = tf.concat(axis=1, values=[net, self.prev_rewards, self.prev_actions])

            # LSTM
            _rnn_out_size = 256
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
            self.a_pred_max = tf.reduce_mean(tf.reduce_max(self.pred, axis=1))
            self.values = tf.squeeze(tf.layers.dense(inputs=net, units=1, name="values"))

            # Loss 계산
            _policy_gain = -tf.reduce_sum(tf.log(self.pred + 1e-5) * self.actions, axis=1) * self.advantages
            _policy_gain = tf.reduce_mean(_policy_gain)
            _entropy = - tf.reduce_sum(self.pred * tf.log(self.pred + 1e-5), axis=1)
            _entropy = tf.reduce_mean(_entropy)
            _value_loss = tf.losses.mean_squared_error(self.values, self.rewards, scope="value_loss")

            self.total_loss = _policy_gain + (_value_loss * 0.5) - (_entropy * 0.01)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            self.gradients = self.optimizer.compute_gradients(self.total_loss, self.var_list)
            self.gradients_placeholders = []

            for grad, var in self.gradients:
                placeholder = tf.placeholder(var.dtype, shape=var.get_shape())
                placeholder = tf.clip_by_norm(placeholder, 10)
                self.gradients_placeholders.append((placeholder, var))
            self.apply_gradients = self.optimizer.apply_gradients(self.gradients_placeholders)

            # Reward list
            self.reward_list = tf.Variable(tf.zeros([self.reward_list_size]), trainable=False, dtype=tf.float32,
                                           name='reward_list')
            self.reward_in = tf.placeholder(dtype=tf.float32, shape=[self.reward_list_size], name='reward')
            self.reward_add_op = self.reward_list.assign(self.reward_in)


class Agent():

    def __init__(self, id, input_shape, output_dim, learning_rate, cluster, is_training=True):
        super(Agent, self).__init__()

        self.id = id

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = gym.make("Pong-v0")

        self.is_training = is_training

        worker_device = "/job:worker/task:%d" % id
        replica_device = tf.train.replica_device_setter(
            worker_device=worker_device,
            ps_device="/job:ps",
            cluster=cluster)

        # Global network
        with tf.device(replica_device):
            self.global_network = A3CNetwork('global', input_shape, output_dim, learning_rate)

            self.global_step = tf.train.get_or_create_global_step()
            self.global_step_add = self.global_step.assign_add(1)

        # Local network
        with tf.device(worker_device):
            self.local_network = A3CNetwork('local', input_shape, output_dim, learning_rate)

        # Weight sync global to local
        self.sync_op = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(self.local_network.var_list, self.global_network.var_list)])

    # State 전처리
    def preProcessState(self, states):
        resize_img = cv2.resize(states, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        return resize_img

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

        rnn_state = self.local_network.state_init

        r = 0
        action_one_hot = np.zeros(self.output_dim)

        while not done:
            episode_step += 1

            # Action 추출
            feed = {
                self.local_network.states: [s],
                self.local_network.state_in[0]: rnn_state[0],
                self.local_network.state_in[1]: rnn_state[1],
                self.local_network.prev_rewards: [[r]],
                self.local_network.prev_actions: [action_one_hot]
            }
            action_prob, v, rnn_state = sess.run([self.local_network.pred, self.local_network.values, self.local_network.state_out],
                                                 feed_dict=feed)
            action_prob = np.squeeze(action_prob)
            a = np.random.choice(np.arange(self.output_dim) + 1, size=1, p=action_prob)[0]

            # 주어진 행동을 실행하고 다음 상태, 보상, 종료 여부를 가져옴
            s1, r, done, _ = self.env.step(a)

            # Action에 one_hot 적용
            action_one_hot = np.squeeze(np.eye(self.output_dim)[a - 1:a])

            # 버퍼에 현재 상태, 행동, 보상, 다음 상태, 종료 여부를 저장한다
            episode_buffer.append([s, action_one_hot, r, v])

            # State 변경
            s1 = self.preProcessState(s1)
            s = s1

            episode_reward += r

            # Episode 학습 - Pong
            if self.is_training:
                if r == -1 or r == 1 or done:
                    if len(episode_buffer) > 2:
                        a_pred_max = self.train(sess, episode_buffer)
                        sess.run(self.sync_op)

                    # Buffer 초기화
                    episode_buffer = []

                    # Rnn state 초기화
                    rnn_state = self.local_network.state_init
            else:
                self.env.render()

        # # Episode 학습 - Breakout
        # if self.is_training:
        #     a_pred_max = self.train(sess, episode_buffer)

        # Episode 보상 출력 및 global 에 기록
        r_list = sess.run(self.global_network.reward_list)
        r_list = np.append(r_list, [episode_reward], axis=0)
        r_list = r_list[1:]

        avg_reward = np.mean(r_list)

        sess.run(self.global_network.reward_add_op, feed_dict={self.global_network.reward_in: r_list})

        # 러닝 시간
        duration = time.time() - start_time
        frame_sec = episode_step / float(duration + 1e-6)

        return step, episode_reward, avg_reward, frame_sec, a_pred_max, episode_step

    # Model 학습
    def train(self, sess, buffer):

        buffer = np.reshape(buffer, [-1, 4])
        xs = np.stack(buffer[:, 0], 0)
        ys = np.vstack(buffer[:, 1])
        train_rewards = buffer[:, 2]
        values = buffer[:, 3]

        prev_r = np.vstack([0] + train_rewards[:-1].tolist())
        prev_a = np.vstack([np.zeros(self.output_dim), ys[:-1].tolist()])

        rnn_state = self.local_network.state_init

        discount_rewards = calc_discount_rewards(train_rewards)
        discount_rewards -= np.mean(discount_rewards)
        discount_rewards /= np.std(discount_rewards) + 1e-8

        advantage = discount_rewards - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-8

        feed = {
            self.local_network.states: xs,
            self.local_network.actions: ys,
            self.local_network.rewards: discount_rewards,
            self.local_network.advantages: advantage,
            self.local_network.state_in[0]: rnn_state[0],
            self.local_network.state_in[1]: rnn_state[1],
            self.local_network.prev_rewards: prev_r,
            self.local_network.prev_actions: prev_a
        }

        gradients, a_pred_max = sess.run([self.local_network.gradients, self.local_network.a_pred_max], feed)

        feed = []
        for (grad, _), (placeholder, _) in zip(gradients, self.global_network.gradients_placeholders):
            feed.append((placeholder, grad))

        feed = dict(feed)
        sess.run(self.global_network.apply_gradients, feed)

        return a_pred_max

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
        print('Learning_rate : ', args.learning_rate)

        # env 환경 파라메터
        input_shape = [105, 80, 3]
        output_dim = 3  # 1, 2, 3

        single_agent = Agent(id=args.task_index,
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

            try:
                # 모델 그래프 최종 확정
                tf.get_default_graph().finalize()

                print('Model training starting...')

                before_step = sess.run(tf.train.get_global_step()) - 1
                before_time = time.time()

                # 100 step 마다 summaray 기록
                div_num = 100
                summary_idx = int(before_step / div_num)

                while not sv.should_stop():
                    step, ep_r, avg_r, frame_sec, a_pred_max, frame_cnt = single_agent.play_episode(sess)

                    message = "Task: {}, Episode: {}, reward= {:.2f}, avg= {:.2f} ({:.2f} frame/sec)".format(args.task_index, step, ep_r, avg_r, frame_sec)
                    print(message)

                    n_step = step - before_step
                    e_time = time.time()
                    step_per_sec = n_step / float((e_time - before_time) + 1e-6)

                    before_step = step
                    before_time = e_time

                    temp_idx = int(step / div_num)

                    if is_chief and summary_idx != temp_idx:

                        summary_idx = temp_idx

                        summary = tf.Summary()
                        summary.value.add(tag='summary/avg_reward', simple_value=float(avg_r))
                        # summary.value.add(tag='summary/a_pred_max', simple_value=float(a_pred_max))
                        summary.value.add(tag='summary/step_sec', simple_value=float(step_per_sec))
                        summary.value.add(tag='summary/frame_count', simple_value=float(frame_cnt))
                        sv.summary_computed(sess, summary, global_step=step)

                    if step >= args.max_steps:
                        sv.request_stop()

            except Exception as ex:  # 에러 종류
                print('에러가 발생 했습니다', ex)
                err_file = os.path.join(args.logdir, "error.log")
                f = open(err_file, 'w')
                f.write(ex.__str__())
                f.close()



# Test용 method
def main_test():
    pass

if __name__ == '__main__':
    if args.mode == 'train':
        main_train()
    else:
        main_test()
