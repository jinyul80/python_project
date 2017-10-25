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
parser.add_argument('--logdir', default='./log/7.1_a3c_pacman_log')
parser.add_argument('--max_steps', default=1000001, type=int)
parser.add_argument('--worker_hosts_num', default=5, type=int)
parser.add_argument('--ps_hosts', type=str, default='localhost:49000',
                    help="Comma-separated list of hostname:port pairs")
parser.add_argument('--job_name', type=str, help="One of 'ps', 'worker'")
parser.add_argument('--task_index', type=int, help="Index of task within the job")

args = parser.parse_args()


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
    def __init__(self, input_size, input_shape, output_size, learning_rate):
        self.input_size = input_size
        self.input_shape = input_shape
        self.output_size = output_size
        self.learning_rate = learning_rate

        self._build()

    def _build(self):
        # 신경망은 게임으로부터 벡터화된 배열로 프레임을 받아서
        # 이것을 리사이즈 하고, 4개의 콘볼루션 레이어를 통해 처리한다.

        # 입력값을 받는 부분 21168 차원은 84*84*3 의 차원이다.
        self.states = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32, name='input_state')
        self.actions = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32, name='input_action')
        self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
        self.rewards = tf.placeholder(tf.float32, [None], name="reward")

        # conv2d 처리를 위해 84x84x3 으로 다시 리사이즈
        _imageIn = tf.reshape(self.states, shape=[-1, *self.input_shape])

        # Dense 레이어
        _dense = tf.layers.dense(inputs=_imageIn, units=128, activation=tf.nn.relu)
        _dense = tf.layers.dense(inputs=_dense, units=64, activation=tf.nn.relu)

        # LSTM
        _rnn_out_size = 32
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

        # Loss 계산
        _policy_gain = -tf.reduce_sum(tf.log(self.pred + 1e-5) * self.actions, axis=1) * self.advantages
        _policy_gain = tf.reduce_mean(_policy_gain)
        _entropy = - tf.reduce_sum(self.pred * tf.log(self.pred + 1e-5), axis=1)
        _entropy = tf.reduce_mean(_entropy)
        _value_loss = tf.losses.mean_squared_error(self.values, self.rewards, scope="value_loss")

        self.total_loss = _policy_gain + (_value_loss * 0.5) - (_entropy * 0.01)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.total_loss)



class Agent():
    global_reward_list = []  # 100개 까지 저장

    def __init__(self, env, id, input_size, input_shape, output_dim, learning_rate, device, is_training=True):
        super(Agent, self).__init__()

        self.id = id

        self.local = A3CNetwork(input_size, input_shape, output_dim, learning_rate)

        self.input_size = input_size
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = env
        self.device = device

        self.is_training = is_training

    # State 전처리
    def preProcessState(self, states):
        return np.reshape(states, [self.input_size])

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

        rnn_state = self.local.state_init

        while not done:
            episode_step += 1

            # Action 추출
            feed = {
                self.local.states: [s],
                self.local.state_in[0]: rnn_state[0],
                self.local.state_in[1]: rnn_state[1]
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
        Agent.global_reward_list.append(episode_reward)
        if len(Agent.global_reward_list) > 100:
            Agent.global_reward_list = Agent.global_reward_list[1:]
        avg_reward = np.mean(Agent.global_reward_list)

        # Traing 모드일 경우 학습
        if self.is_training:
            self.train(sess, episode_buffer)

        # 러닝 시간
        duration = time.time() - start_time
        sec_per_step = float(duration) / episode_step

        return episode_reward, avg_reward, sec_per_step

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

        _ = sess.run(self.local.train_op, feed)


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
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    if args.job_name == "ps":
        # Create and start a server for the local task.
        server = tf.train.Server(cluster,
                                 job_name=args.job_name,
                                 task_index=args.task_index,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))

        print('Parameter server starting...')
        server.join()

    elif args.job_name == "worker":
        # Create and start a server for the local task.
        server = tf.train.Server(cluster,
                                 job_name=args.job_name,
                                 task_index=args.task_index)

        print('Worker server starting...')
        device = tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % args.task_index,
            ps_device="/job:ps",
            cluster=cluster)

        with tf.device(device):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            global_step_op = global_step.assign_add(1)

            # env 환경 파라메터
            input_size = 60
            input_shape = [60]
            output_dim = 3

            env = mini_pacman.Gym(show_game=False)

            single_agent = Agent(env=env,
                                 id=args.task_index,
                                 input_size=input_size,
                                 input_shape=input_shape,
                                 output_dim=output_dim,
                                 learning_rate=args.learning_rate,
                                 device=device)

            saver = tf.train.Saver()
            init_op = tf.global_variables_initializer()

        is_chief = (args.task_index == 0)
        # 훈련 과정을 살펴보기 위해 "supervisor"를 생성한다.
        # summary와 placeholder 에러 발생으로 summary_op는 None 처리.
        # 차후 일정 주기마다 수동으로 summary 실행
        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir=args.logdir,
                                 init_op=init_op,
                                 summary_op=None,
                                 saver=saver,
                                 save_model_secs=0,
                                 global_step=global_step)

        # supervisor는 세션 초기화를 관리하고, checkpoint로부터 모델을 복원하고
        # 에러가 발생하거나 연산이 완료되면 프로그램을 종료한다.
        sess_config = tf.ConfigProto(
            log_device_placement=False,
            device_filters=["/job:ps", "/job:worker/task:%d" % args.task_index])

        fName = os.path.join(args.logdir, "model.ckpt")
        saveIdx = 0

        with sv.managed_session(server.target, config=sess_config) as sess:
            # "supervisor"가 종료되거나 MAX_LOOP step이 수행 될 때까지 반복한다.
            local_step = 0
            step = 0

            if is_chief:
                # Model restore
                ckpt = tf.train.get_checkpoint_state(args.logdir)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    step = sess.run(global_step)

                    print('model restore... current step :', step)

            while not sv.should_stop() and step < args.max_steps:
                # 훈련 과정을 비동기식으로 실행한다.Run a training step asynchronously.
                # 동기식 훈련 수행을 위해서는 `tf.train.SyncReplicasOptimizer`를 참조하라.

                ep_r, avg_r, ep_sec = single_agent.play_episode(sess)
                step = sess.run(global_step_op)
                message = "Task: {}, Episode: {}, reward= {:.2f}, avg= {:.2f} ({:.5f} sec)".format(args.task_index, step, ep_r, avg_r, ep_sec)
                print(message)

                if is_chief and step != 0:
                    summary = tf.Summary()
                    summary.value.add(tag='Reward_avg', simple_value=float(avg_r))
                    sv.summary_computed(sess, summary, global_step=step)

                    # Global step 기준으로 1000 step마다 model save
                    if saveIdx != int(step / 1000):
                        sv.saver.save(sess, fName, global_step=step)
                        saveIdx = int(step / 1000)
                        print('model saved...')

                # Local step 증가
                local_step += 1
                
                # 평균 점수 300 이상이면 종료
                if avg_r > 300:
                    break

            # Loop 종료 시 저장
            if is_chief:
                sv.saver.save(sess, fName, global_step=step)

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
