import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import threading
import argparse
import time
import queue
from gridworld import gameEnv

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Runner thread와 Train thread 분리
# 테스트 결과 Runner thread에서 학습까지 시키는 것 보다 느림.

parser = argparse.ArgumentParser(description="Simple 'argparse' demo application")
parser.add_argument('--mode', default='train', help='Execute mode')
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--logdir', default='./log/4.4_a3c_gridworld_log')
parser.add_argument('--max_steps', default=1000001, type=int)
parser.add_argument('--n_threads', default=2, type=int)
parser.add_argument('--env_size', default=5, type=int)
parser.add_argument('--max_ep_steps', default=50, type=int)

args = parser.parse_args()


# 학습 변수 copy 함수
def copy_src_to_dst(from_scope, to_scope):
    """Creates a copy variable weights operation

    Args:
        from_scope (str): The name of scope to copy from
            It should be "global"
        to_scope (str): The name of scope to copy to
            It should be "thread-{}"

    Returns:
        list: Each element is a copy operation
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


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
    def __init__(self, name, input_size, input_shape, output_size):
        self.name = name
        self.input_size = input_size
        self.input_shape = input_shape
        self.output_size = output_size

        self._build()

        self.global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)

    def _build(self):
        with tf.variable_scope(self.name):
            # 신경망은 게임으로부터 벡터화된배열로 프레임을 받아서
            # 이것을 리사이즈 하고, 4개의 콘볼루션 레이어를 통해 처리한다.

            # 입력값을 받는 부분 21168 차원은 84*84*3 의 차원이다.
            self.states = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32, name='input_state')
            self.actions = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32, name='input_action')
            self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
            self.rewards = tf.placeholder(tf.float32, [None], name="reward")

            # conv2d 처리를 위해 84x84x3 으로 다시 리사이즈6
            _imageIn = tf.reshape(self.states, shape=[-1, *self.input_shape])

            # 콘볼루션을 통해 이미지 인코딩
            _conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=_imageIn, num_outputs=32,
                                 kernel_size=[3, 3], stride=[2, 2], padding='VALID')
            _conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=_conv1, num_outputs=64,
                                 kernel_size=[3, 3], stride=[2, 2], padding='VALID')
            _conv3 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=_conv2, num_outputs=128,
                                 kernel_size=[3, 3], stride=[2, 2], padding='VALID')
            net = slim.conv2d(activation_fn=tf.nn.elu,
                              inputs=_conv3, num_outputs=256,
                              kernel_size=[3, 3], stride=[2, 2], padding='VALID')

            # Inception block
            # output : [4, 4, 480]
            with tf.variable_scope('inception1'):
                branch_0 = tf.layers.conv2d(inputs=net, filters=128, kernel_size=[1, 1], padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation=tf.nn.relu)
                branch_1 = tf.layers.conv2d(inputs=net, filters=128, kernel_size=[1, 1], padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation=tf.nn.relu)
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=192, kernel_size=[3, 3], padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation=tf.nn.relu)
                branch_2 = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[1, 1], padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation=tf.nn.relu)
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=64, kernel_size=[3, 3], padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation=tf.nn.relu)
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=96, kernel_size=[3, 3], padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation=tf.nn.relu)
                branch_3 = tf.layers.average_pooling2d(inputs=net, pool_size=[3, 3], strides=1, padding='same')
                branch_3 = tf.layers.conv2d(inputs=branch_3, filters=64, kernel_size=[1, 1], padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation=tf.nn.relu)
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            # Inception block
            # output : [4, 4, 832]
            with tf.variable_scope('inception2'):
                branch_0 = tf.layers.conv2d(inputs=net, filters=256, kernel_size=[1, 1], padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation=tf.nn.relu)
                branch_1 = tf.layers.conv2d(inputs=net, filters=160, kernel_size=[1, 1], padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation=tf.nn.relu)
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=320, kernel_size=[3, 3], padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation=tf.nn.relu)
                branch_2 = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[1, 1], padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation=tf.nn.relu)
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=128, kernel_size=[1, 3], padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation=tf.nn.relu)
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=128, kernel_size=[3, 1], padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation=tf.nn.relu)
                branch_3 = tf.layers.average_pooling2d(inputs=net, pool_size=[3, 3], strides=1, padding='same')
                branch_3 = tf.layers.conv2d(inputs=branch_3, filters=128, kernel_size=[1, 1], padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation=tf.nn.relu)
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            net = tf.contrib.layers.flatten(net)
            _dense = slim.fully_connected(net, 1024, activation_fn=tf.nn.elu)

            # LSTM
            _rnn_out_size = 512
            _rnn_in = tf.expand_dims(_dense, [0])
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
            rnn_out = tf.reshape(lstm_outputs, [-1, _rnn_out_size])

            # Output
            self.pred = tf.layers.dense(inputs=rnn_out, units=self.output_size, activation=tf.nn.softmax, name='pred')
            self.values = tf.squeeze(tf.layers.dense(inputs=rnn_out, units=1, name="values"))


# Runner agent class
class RunnerAgent(threading.Thread):
    global_episode = 1
    max_ep_steps = 50  # episode별 최대 걸음 수

    def __init__(self, session, env, coord, id, input_size, input_shape, output_dim, data_q, is_training=True):
        super(RunnerAgent, self).__init__()

        self.id = id
        self.name = "thread_{}".format(id)

        self.local = A3CNetwork(self.name, input_size, input_shape, output_dim)
        self.global_to_local = copy_src_to_dst("global", self.name)

        self.input_size = input_size
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = env
        self.sess = session
        self.coord = coord

        self.data_q = data_q
        self.is_training = is_training

        self.lock = threading.Lock()

    # State 전처리
    def preProcessState(self, states):
        return np.reshape(states, [self.input_size])

    def run(self):
        while not self.coord.should_stop():

            # Queue가 가득차면 10초 대기
            while True:
                if self.data_q.qsize() > 50:
                    time.sleep(10)
                else:
                    break

            self.play_episode()

    def play_episode(self):
        local_episode = RunnerAgent.global_episode

        with self.lock:
            RunnerAgent.global_episode += 1

        # Set episode start time
        start_time = time.time()

        # Global 변수를 local model에 복사
        self.sess.run(self.global_to_local)

        s = self.env.reset()
        s = self.preProcessState(s)

        done = False
        episode_reward = 0
        episode_step = 0

        # 에피스드 임시 버퍼
        episode_buffer = []

        rnn_state = self.local.state_init

        while not done:
            episode_step += 1

            # Episode별 XX회 이상되면 종료
            if episode_step > RunnerAgent.max_ep_steps:
                break

            # Action 추출
            feed = {
                self.local.states: [s],
                self.local.state_in[0]: rnn_state[0],
                self.local.state_in[1]: rnn_state[1]
            }
            action_prob, v, rnn_state = self.sess.run([self.local.pred, self.local.values, self.local.state_out],
                                                      feed_dict=feed)
            action_prob = np.squeeze(action_prob)
            a = np.random.choice(self.output_dim, size=1, p=action_prob)[0]

            # 주어진 행동을 실행하고 다음 상태, 보상, 종료 여부를 가져옴
            s1, r, done = self.env.step(a)

            # Action에 one_hot 적용
            action_one_hot = np.squeeze(np.eye(self.output_dim)[a:a + 1])

            # 버퍼에 현재 상태, 행동, 보상, value 를 저장한다
            episode_buffer.append([s, action_one_hot, r, v])

            # State 변경
            s1 = self.preProcessState(s1)
            s = s1

            episode_reward += r

        if self.is_training:
            # Episode 종료 시 Queue에 저장
            self.data_q.put([local_episode, episode_buffer, episode_reward])

        # 러닝 시간
        duration = time.time() - start_time
        sec_per_step = float(duration)

        # print('Episode {}, Agent(name={} reward : {}) ({:.2f} sec)'.format(local_episode, self.id, episode_reward,
        #                                                                    sec_per_step))


class TrainAgent():
    def __init__(self, session, coord, global_network, learning_rate, data_q, logdir=None):
        super(TrainAgent, self).__init__()

        self.id = id
        self.name = 'global'

        self.global_network = global_network

        self.sess = session
        self.coord = coord
        self.data_q = data_q

        self.logdir = logdir

        # Loss 계산
        _policy_gain = -tf.reduce_sum(tf.log(self.global_network.pred + 1e-5) * self.global_network.actions, axis=1)
        _policy_gain *= self.global_network.advantages
        _policy_gain = tf.reduce_mean(_policy_gain)
        _entropy = - tf.reduce_sum(self.global_network.pred * tf.log(self.global_network.pred + 1e-5), axis=1)
        _entropy = tf.reduce_mean(_entropy)
        _value_loss = tf.losses.mean_squared_error(self.global_network.values, self.global_network.rewards)

        total_loss = _policy_gain + (_value_loss * 0.5) - (_entropy * 0.01)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.train_op = self.optimizer.minimize(total_loss)

        if self.logdir:
            self.reward = tf.placeholder(tf.float32, name="reward")
            tf.summary.scalar("reward", self.reward)

            self.summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.logdir)

    # Model 학습
    def train(self, step, buffer, reward):
        # Set episode start time
        start_time = time.time()

        buffer = np.reshape(buffer, [-1, 4])
        xs = np.vstack(buffer[:, 0])
        ys = np.vstack(buffer[:, 1])
        train_rewards = buffer[:, 2]
        values = buffer[:, 3]

        discount_rewards = calc_discount_rewards(train_rewards)
        discount_rewards -= np.mean(discount_rewards)
        discount_rewards /= np.std(discount_rewards) + 1e-8

        advantage = discount_rewards - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-8

        rnn_state = self.global_network.state_init

        feed = {
            self.global_network.states: xs,
            self.global_network.actions: ys,
            self.global_network.rewards: discount_rewards,
            self.global_network.advantages: advantage,
            self.global_network.state_in[0]: rnn_state[0],
            self.global_network.state_in[1]: rnn_state[1],
            self.reward: reward
        }

        _, summary = self.sess.run([self.train_op, self.summary_op], feed)
        self.summary_writer.add_summary(summary, global_step=step)

        # 러닝 시간
        duration = time.time() - start_time
        sec_per_step = float(duration)

        print('Trained Episode : {}'.format(step), 'reward :', reward,
              'Queue size :', self.data_q.qsize(), '({:.2f} sec)'.format(sec_per_step))


# 학습용 method
def main_train():
    try:
        tf.reset_default_graph()

        sess = tf.Session()
        coord = tf.train.Coordinator()

        checkpoint_dir = args.logdir
        save_path = os.path.join(checkpoint_dir, "model.ckpt")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print("Directory {} was created".format(checkpoint_dir))

        # env 환경 파라메터
        input_size = 21168
        input_shape = [84, 84, 3]
        output_dim = 4
        global_network = A3CNetwork(name="global",
                                    input_size=input_size,
                                    input_shape=input_shape,
                                    output_size=output_dim)
        thread_list = []
        env_list = []

        data_q = queue.Queue()

        for id in range(args.n_threads):
            env = gameEnv(partial=False, size=args.env_size)

            run_agent = RunnerAgent(env=env,
                                    session=sess,
                                    coord=coord,
                                    id=id,
                                    input_size=input_size,
                                    input_shape=input_shape,
                                    output_dim=output_dim,
                                    data_q=data_q,
                                    is_training=True)
            run_agent.max_ep_steps = args.max_ep_steps

            thread_list.append(run_agent)
            env_list.append(env)

        train_agent = TrainAgent(session=sess,
                                 coord=coord,
                                 global_network=global_network,
                                 learning_rate=args.learning_rate,
                                 data_q=data_q,
                                 logdir=args.logdir)

        # 모델 초기화
        init = tf.global_variables_initializer()
        sess.run(init)

        train_agent.summary_writer.add_graph(sess.graph)

        # Save 파일 있을 경우 복구
        save_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
        saver = tf.train.Saver(var_list=save_var_list)

        if tf.train.get_checkpoint_state(os.path.dirname(save_path)):
            saver.restore(sess, save_path)
            RunnerAgent.global_episode = sess.run(global_network.global_episodes)
            print("Model restored to global")

        else:
            print("No model is found")

        print("Runner start...")
        print("Ctrl + C to close")

        # Episode runner start
        for t in thread_list:
            t.start()
            time.sleep(1)

        # Trainer start
        while True:
            step, buffer, reward = data_q.get(timeout=120.0)

            sess.run(global_network.global_episodes.assign(step))
            train_agent.train(step, buffer, reward)

            if step % 1000 == 0:
                saver.save(sess, save_path)
                print('Checkpoint Saved to {}'.format(save_path))

            # 최대 episode 이상 이면 종료
            if step >= args.max_steps:
                if not coord.should_stop():
                    print("Closing threads")
                    coord.request_stop()
                    coord.join(thread_list)
                break

    except KeyboardInterrupt:
        print("Closing threads")
        coord.request_stop()
        coord.join(thread_list)

        print("Closing environments")
        for env in env_list:
            try:
                env.close()
            except:
                pass

    finally:
        saver.save(sess, save_path)
        print('Checkpoint Saved to {}'.format(save_path))

    sess.close()


# Test용 method
def main_test():
    tf.reset_default_graph()

    sess = tf.Session()

    checkpoint_dir = args.logdir
    save_path = os.path.join(checkpoint_dir, "model.ckpt")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print("Directory {} was created".format(checkpoint_dir))

    # env 환경 파라메터
    input_size = 21168
    input_shape = [84, 84, 3]
    output_dim = 4
    global_network = A3CNetwork(name="global",
                                input_size=input_size,
                                input_shape=input_shape,
                                output_size=output_dim)

    data_q = queue.Queue()

    env = gameEnv(partial=False, size=args.env_size)
    env.show_game = True

    single_agent = RunnerAgent(env=env,
                               session=sess,
                               coord=None,
                               id=0,
                               input_size=input_size,
                               input_shape=input_shape,
                               output_dim=output_dim,
                               data_q=data_q,
                               is_training=False)

    # 모델 초기화
    init = tf.global_variables_initializer()
    sess.run(init)

    save_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
    saver = tf.train.Saver(var_list=save_var_list)

    # Save 파일 있을 경우 복구
    if tf.train.get_checkpoint_state(os.path.dirname(save_path)):
        saver.restore(sess, save_path)
        print("Model restored to global")
    else:
        print("No model is found")

    single_agent.play_episode()

    sess.close()


if __name__ == '__main__':
    if args.mode == 'train':
        main_train()
    else:
        main_test()
