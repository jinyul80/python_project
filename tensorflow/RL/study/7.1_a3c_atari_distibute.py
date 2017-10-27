import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import argparse
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description="Simple 'argparse' demo application")
parser.add_argument('--mode', default='train', help='Execute mode')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--logdir', default='./log/7.1_a3c_atari_log')
parser.add_argument('--max_steps', default=1000001, type=int)
parser.add_argument('--worker_hosts_num', default=5, type=int)
parser.add_argument('--ps_hosts', type=str, default='localhost:49000',
                    help="Comma-separated list of hostname:port pairs")
parser.add_argument('--job_name', type=str, help="One of 'ps', 'worker'")
parser.add_argument('--task_index', type=int, default=0, help="Index of task within the job")

args = parser.parse_args()


def cluster_spec(num_workers, num_ps):
    """
    Tensorflow 분산 환경 설정
    """
    cluster = {}
    port = 49000

    all_ps = []
    host = '192.168.1.122'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    # PC1
    port = 49100
    host = '192.168.1.122'
    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1

    # PC2
    host = '192.168.1.104'
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


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
        http://karpathy.github.io/2016/05/31/rl/ """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


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

            # conv2d 처리를 위해 84x84x3 으로 다시 리사이즈
            _imageIn = tf.reshape(self.states, shape=[-1, *self.input_shape])

            # 콘볼루션을 통해 이미지 인코딩
            _conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=_imageIn, num_outputs=16,
                                 kernel_size=[8, 8], stride=[4, 4], padding='VALID')
            net = slim.conv2d(activation_fn=tf.nn.elu,
                              inputs=_conv1, num_outputs=32,
                              kernel_size=[4, 4], stride=[2, 2], padding='VALID')

            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], stride=2)

            # Inception block
            # output : [4, 4, 128]
            with tf.variable_scope('inception1'):
                branch_0 = slim.conv2d(inputs=net, num_outputs=16, kernel_size=[1, 1],
                                       activation_fn=tf.nn.elu, padding='SAME')
                branch_1 = slim.conv2d(inputs=net, num_outputs=16, kernel_size=[1, 1],
                                       activation_fn=tf.nn.elu, padding='SAME')
                branch_1 = slim.conv2d(inputs=branch_1, num_outputs=32, kernel_size=[3, 3],
                                       activation_fn=tf.nn.elu, padding='SAME')
                branch_2 = slim.conv2d(inputs=net, num_outputs=16, kernel_size=[1, 1],
                                       activation_fn=tf.nn.elu, padding='SAME')
                branch_2 = slim.conv2d(inputs=branch_2, num_outputs=32, kernel_size=[3, 3],
                                       activation_fn=tf.nn.elu, padding='SAME')
                branch_2 = slim.conv2d(inputs=branch_2, num_outputs=32, kernel_size=[3, 3],
                                       activation_fn=tf.nn.elu, padding='SAME')
                branch_3 = slim.avg_pool2d(inputs=net, kernel_size=[3, 3], stride=1, padding='SAME')
                branch_3 = slim.conv2d(inputs=branch_3, num_outputs=48, kernel_size=[1, 1],
                                       activation_fn=tf.nn.elu, padding='SAME')

                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            net = tf.contrib.layers.flatten(net)
            _dense1 = slim.fully_connected(net, 512, activation_fn=tf.nn.elu)

            # LSTM
            _rnn_out_size = 64
            _rnn_in = tf.expand_dims(_dense1, [0])
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

            if self.name != 'global':
                # Loss 계산
                _policy_gain = -tf.reduce_sum(tf.log(self.pred + 1e-5) * self.actions, axis=1) * self.advantages
                _policy_gain = tf.reduce_mean(_policy_gain)
                _entropy = - tf.reduce_sum(self.pred * tf.log(self.pred + 1e-5), axis=1)
                _entropy = tf.reduce_mean(_entropy)
                _value_loss = tf.losses.mean_squared_error(self.values, self.rewards, scope="value_loss")

                self.total_loss = _policy_gain + (_value_loss * 0.5) - (_entropy * 0.01)

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Agent():
    global_reward_list = []  # 100개 까지 저장

    def __init__(self, env, id, input_size, input_shape, output_dim, learning_rate, cluster, is_training=True):
        super(Agent, self).__init__()

        self.id = id

        self.input_size = input_size
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = env

        self.is_training = is_training

        worker_device = "/job:worker/task:%d" % id
        replica_device = tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % id,
            ps_device="/job:ps",
            cluster=cluster)

        # Global network
        with tf.device(replica_device):
            self.global_network = A3CNetwork('global', input_size, input_shape, output_dim, learning_rate)
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.global_step_op = self.global_step.assign_add(1)

        # Local network
        with tf.device(worker_device):
            self.local = A3CNetwork('local', input_size, input_shape, output_dim, learning_rate)

        # Weight sync global to local
        self.sync_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.local.var_list, self.global_network.var_list)])

        # Train op
        grads = tf.gradients(self.local.total_loss, self.local.var_list)
        grads, _ = tf.clip_by_global_norm(grads, 40.0)

        grads_and_vars = list(zip(grads, self.global_network.var_list))

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = opt.apply_gradients(grads_and_vars)

    # State 전처리
    def preProcessState(self, states):
        return np.reshape(states, [self.input_size])

    def play_episode(self, sess):
        # Set episode start time
        start_time = time.time()

        s = self.env.reset()
        s = prepro(s)

        done = False
        episode_reward = 0

        # 에피스드 임시 버퍼
        episode_buffer = []

        action_count = [0, 0, 0]

        rnn_state = self.local.state_init

        while not done:
            # Action 추출
            feed = {
                self.local.states: [s],
                self.local.state_in[0]: rnn_state[0],
                self.local.state_in[1]: rnn_state[1]
            }
            action_prob, v, rnn_state = sess.run([self.local.pred, self.local.values, self.local.state_out],
                                                 feed_dict=feed)
            action_prob = np.squeeze(action_prob)
            a = np.random.choice(np.arange(self.output_dim) + 1, size=1, p=action_prob)[0]

            action_count[a - 1] += 1

            # 주어진 행동을 실행하고 다음 상태, 보상, 종료 여부를 가져옴
            s1, r, done, _ = self.env.step(a)

            # Action에 one_hot 적용
            action_one_hot = np.squeeze(np.eye(self.output_dim)[a - 1:a])

            # 버퍼에 현재 상태, 행동, 보상, 다음 상태, 종료 여부를 저장한다
            episode_buffer.append([s, action_one_hot, r, v])

            # State 변경
            s1 = prepro(s1)
            s = s1

            episode_reward += r

            # Episode 학습
            if self.is_training:
                if r == -1 or r == 1 or done:
                    self.train(sess, episode_buffer)
                    sess.run(self.sync_op)

                    # Buffer 초기화
                    episode_buffer = []

                    # Rnn state 초기화
                    rnn_state = self.local.state_init

            else:
                self.env.render()

        # Episode 보상 출력 및 global에 기록
        Agent.global_reward_list.append(episode_reward)
        if len(Agent.global_reward_list) > 100:
            Agent.global_reward_list = Agent.global_reward_list[1:]
        avg_reward = np.mean(Agent.global_reward_list)

        # 러닝 시간
        duration = time.time() - start_time
        sec_per_step = float(duration)

        return episode_reward, avg_reward, sec_per_step, action_count

    # Model 학습
    def train(self, sess, buffer):

        buffer = np.reshape(buffer, [-1, 4])
        xs = np.vstack(buffer[:, 0])
        ys = np.vstack(buffer[:, 1])
        train_rewards = buffer[:, 2]
        values = buffer[:, 3]

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
        }

        _ = sess.run(self.train_op, feed)


# 학습용 method
def main_train():

    ps_hosts = args.ps_hosts.split(",")
    worker_hosts = []

    port_num = 49100

    for i in range(args.worker_hosts_num):
        worker_hosts.append("localhost:" + str(port_num))
        port_num += 1

    ps_hosts = list(ps_hosts)
    worker_hosts = list(worker_hosts)

    # Create a cluster from the parameter server and worker hosts.

    # # Local
    # cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

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
        input_size = 6400
        input_shape = [80, 80, 1]
        output_dim = 3  # 1, 2, 3

        env = gym.make("Pong-v0")

        single_agent = Agent(env=env,
                             id=args.task_index,
                             input_size=input_size,
                             input_shape=input_shape,
                             output_dim=output_dim,
                             learning_rate=args.learning_rate,
                             cluster=cluster)

        variables_to_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global')
        saver = tf.train.Saver(var_list=variables_to_save)

        init_op = tf.global_variables_initializer()

        local_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='local')
        local_init_op = tf.variables_initializer(local_var_list)

        is_chief = (args.task_index == 0)
        # 훈련 과정을 살펴보기 위해 "supervisor"를 생성한다.
        # summary와 placeholder 에러 발생으로 summary_op는 None 처리.
        # 차후 일정 주기마다 수동으로 summary 실행
        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir=args.logdir,
                                 init_op=init_op,
                                 local_init_op=local_init_op,
                                 summary_op=None,
                                 saver=saver,
                                 save_model_secs=0,
                                 global_step=single_agent.global_step)

        # supervisor는 세션 초기화를 관리하고, checkpoint로부터 모델을 복원하고
        # 에러가 발생하거나 연산이 완료되면 프로그램을 종료한다.
        sess_config = tf.ConfigProto(
            log_device_placement=False,
            device_filters=["/job:ps", "/job:worker/task:%d" % args.task_index])

        fName = os.path.join(args.logdir, "model.ckpt")
        saveIdx = 0

        with sv.prepare_or_wait_for_session(server.target, config=sess_config) as sess:
            # "supervisor"가 종료되거나 MAX_LOOP step이 수행 될 때까지 반복한다.
            local_step = 0
            step = 0

            if is_chief:
                # Model restore
                ckpt = tf.train.get_checkpoint_state(args.logdir)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    step = sess.run(single_agent.global_step)
                    saveIdx = int(step / 1000)

                    print('model restore... current step :', step)

            while not sv.should_stop() and step < args.max_steps:
                # 학습 변수 copy
                sess.run(single_agent.sync_op)

                ep_r, avg_r, ep_sec, ac = single_agent.play_episode(sess)
                step = sess.run(single_agent.global_step_op)
                message = "Task: {}, Episode: {}, reward= {:.2f}, avg= {:.2f} ({:.3f} sec)".format(args.task_index, step, ep_r, avg_r, ep_sec)
                print(message)

                print('Action count : ', np.sum(ac), 'frame/sec: {:.4f}'.format(ep_sec/np.sum(ac)), ':', ac)

                if is_chief and step != 0:
                    summary = tf.Summary()
                    summary.value.add(tag='Reward_avg', simple_value=float(avg_r))
                    sv.summary_computed(sess, summary, global_step=step)

                    # Global step 기준으로 1000 step마다 model save
                    if saveIdx != int(step / 1000):
                        sv.saver.save(sess, fName)
                        saveIdx = int(step / 1000)
                        print('model saved...')

                # Local step 증가
                local_step += 1

            # Loop 종료 시 저장
            if is_chief:
                sv.saver.save(sess, fName)

            sv.stop()
            print('Done')


# Test용 method
def main_test():
    pass

if __name__ == '__main__':
    if args.mode == 'train':
        main_train()
    else:
        main_test()
