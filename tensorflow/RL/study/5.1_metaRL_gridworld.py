import numpy as np
import tensorflow as tf
import os
import threading
import argparse
import time
from gridworld import gameEnv
import cv2

parser = argparse.ArgumentParser(description="Simple 'argparse' demo application")
parser.add_argument('--mode', default='train', help='Execute mode')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--logdir', default='./log/gridworld_log/metaRL_1e-4')
parser.add_argument('--max_steps', default=100001, type=int)
parser.add_argument('--n_threads', default=4, type=int)
parser.add_argument('--update_ep_size', default=1, type=int)
parser.add_argument('--env_size', default=5, type=int)
parser.add_argument('--max_ep_steps', default=50, type=int)

args = parser.parse_args()

# GPU not use
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# env 환경 파라메터
input_size = 42 * 42 * 3
input_shape = [42, 42, 3]
output_dim = 4


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
    def __init__(self, name, input_size, input_shape, output_size, learning_rate, logdir=None):
        self.name = name
        self.input_size = input_size
        self.input_shape = input_shape
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.logdir = logdir

        self._build()

        self.global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        self.input_ep = tf.placeholder(dtype=tf.int32)
        self.update_ep = self.global_episodes.assign(self.input_ep)

    def _build(self):
        with tf.variable_scope(self.name):
            # 신경망은 게임으로부터 벡터화된배열로 프레임을 받아서
            # 이것을 리사이즈 하고, 4개의 콘볼루션 레이어를 통해 처리한다.

            # 입력값을 받는 부분 5292 차원은 42*42*3 의 차원이다.
            self.states = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32, name='input_state')
            self.actions = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32, name='input_action')
            self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
            self.rewards = tf.placeholder(tf.float32, [None], name="reward")

            self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='prev_rewards')
            self.prev_actions = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32, name='prev_actions')

            # conv2d 처리를 위해 42x42x3 으로 다시 리사이즈
            _imageIn = tf.reshape(self.states, shape=[-1, *self.input_shape])

            # 콘볼루션을 통해 이미지 인코딩
            with tf.variable_scope('Conv'):
                net = tf.layers.conv2d(inputs=_imageIn, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='same',
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

            # Inception block
            # output : [3, 3, 416]
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

            net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=1)
            net = tf.contrib.layers.flatten(net)
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
            c_init = np.zeros((1, lstm_cell.state_size.c), np.dtype(float))
            # 은닉층 초기화
            h_init = np.zeros((1, lstm_cell.state_size.h), np.dtype(float))
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
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            if self.logdir:
                tf.summary.scalar("a_pred_max", tf.reduce_mean(tf.reduce_max(self.pred, axis=1)))
                # tf.summary.scalar("policy_loss", _policy_gain)
                # tf.summary.scalar("entropy_loss", _entropy)
                # tf.summary.scalar("value_loss", _value_loss)
                tf.summary.scalar("total_loss", self.total_loss)
                tf.summary.histogram("values", self.values)
                tf.summary.histogram("pred", self.pred)

                self.reward_avg = tf.placeholder(tf.float32, name="reward_avg")
                tf.summary.scalar("reward_avg", self.reward_avg)

                self.summary_op = tf.summary.merge_all()
                self.summary_writer = tf.summary.FileWriter(self.logdir)

            if self.name != 'global':
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
                self.gradients = tf.gradients(self.total_loss, var_list)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # 전역 신경망에 적용
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = self.optimizer.apply_gradients(zip(grads, global_vars))


class Agent(threading.Thread):
    global_episode = 1
    global_reward_list = []  # 100개 까지 저장
    max_ep_steps = 50  # episode별 최대 걸음 수

    def __init__(self, session, env, coord, id, global_network, input_size, input_shape, output_dim, learning_rate,
                 update_ep_size, is_training=True, logdir=None):
        super(Agent, self).__init__()

        self.id = id
        self.name = "thread_{}".format(id)

        self.local = A3CNetwork(self.name, input_size, input_shape, output_dim, learning_rate, logdir)
        self.global_to_local = copy_src_to_dst("global", self.name)
        self.global_network = global_network

        self.input_size = input_size
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = env
        self.sess = session
        self.coord = coord

        self.logdir = logdir
        self.update_ep_size = update_ep_size
        self.is_training = is_training

    # Episode 결과 출력
    def print(self, reward, avg_reward, frame_sec):
        message = "Episode : {} , Agent(name={}, reward= {:.2f}, avg= {:.2f}) ({:.2f} frame/sec)".format(self.local_episode,
                                                                                                   self.name, reward,
                                                                                                   avg_reward, frame_sec)
        print(message)

    # State 전처리
    def preProcessState(self, states):
        resize_img = cv2.resize(states, (42, 42))
        return np.reshape(resize_img, [self.input_size])

    def run(self):
        while not self.coord.should_stop():
            self.play_episode()

    def play_episode(self):
        self.local_episode = Agent.global_episode
        Agent.global_episode += 1

        # Set episode start time
        start_time = time.time()

        # Global 변수를 local model에 복사
        self.sess.run(self.global_to_local)

        # N번의 보상을 저장할 리스트
        reward_list = []

        # N번의 episode 기록을 모와서 학습
        for _ in range(self.update_ep_size):
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

                # Episode별 XX회 이상되면 종료
                if episode_step > Agent.max_ep_steps:
                    break

                # Action 추출
                feed = {
                    self.local.states: [s],
                    self.local.state_in[0]: rnn_state[0],
                    self.local.state_in[1]: rnn_state[1],
                    self.local.prev_rewards: [[r]],
                    self.local.prev_actions: [action_one_hot]
                }
                action_prob, v, rnn_state = self.sess.run([self.local.pred, self.local.values, self.local.state_out],
                                                          feed_dict=feed)
                action_prob = np.squeeze(action_prob)
                a = np.random.choice(self.output_dim, size=1, p=action_prob)[0]

                # 주어진 행동을 실행하고 다음 상태, 보상, 종료 여부를 가져옴
                s1, r, done = self.env.step(a)

                # Action에 one_hot 적용
                action_one_hot = np.squeeze(np.eye(self.output_dim)[a:a + 1])

                # 버퍼에 현재 상태, 행동, 보상, 다음 상태, 종료 여부를 저장한다
                episode_buffer.append([s, action_one_hot, r, v])

                # State 변경
                s1 = self.preProcessState(s1)
                s = s1

                episode_reward += r

            # 각 episode 보상을 저장
            reward_list.append(episode_reward)

        # Episode 보상 출력 및 global에 기록
        Agent.global_reward_list.append(np.mean(reward_list))
        if len(Agent.global_reward_list) > 100:
            Agent.global_reward_list = Agent.global_reward_list[1:]
        avg_reward = np.mean(Agent.global_reward_list)

        # N번의 Episode 종료 시 학습
        if self.is_training:
            self.train(episode_buffer)

        # 러닝 시간
        duration = time.time() - start_time
        frame_sec = episode_step / float(duration + 1e-6)

        self.print(np.mean(reward_list), avg_reward, frame_sec)

    # Model 학습
    def train(self, buffer):

        buffer = np.reshape(buffer, [-1, 4])
        xs = np.vstack(buffer[:, 0])
        ys = np.vstack(buffer[:, 1])
        train_rewards = buffer[:, 2]
        values = buffer[:, 3]

        prev_r = np.vstack([0] + train_rewards[:-1].tolist())
        prev_a = np.vstack([np.zeros(self.output_dim), ys[:-1].tolist()])

        discount_rewards = calc_discount_rewards(train_rewards)
        discount_rewards -= np.mean(discount_rewards)
        discount_rewards /= np.std(discount_rewards) + 1e-8

        advantage = discount_rewards - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-8

        rnn_state = self.local.state_init

        feed = {
            self.local.states: xs,
            self.local.actions: ys,
            self.local.rewards: discount_rewards,
            self.local.advantages: advantage,
            self.local.state_in[0]: rnn_state[0],
            self.local.state_in[1]: rnn_state[1],
            self.local.prev_rewards: prev_r,
            self.local.prev_actions: prev_a,
            self.local.reward_avg: np.mean(Agent.global_reward_list)
        }

        if self.id == 0:
            summ, _ = self.sess.run([self.local.summary_op, self.local.apply_grads], feed)
            self.local.summary_writer.add_summary(summ, global_step=self.local_episode)
        else:
            self.sess.run(self.local.apply_grads, feed)


# 학습용 method
def main_train():
    try:
        tf.reset_default_graph()

        sess = tf.InteractiveSession()
        coord = tf.train.Coordinator()

        checkpoint_dir = args.logdir
        save_path = os.path.join(checkpoint_dir, "model.ckpt")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print("Directory {} was created".format(checkpoint_dir))

        # Global network 생성
        global_network = A3CNetwork(name="global",
                                    input_size=input_size,
                                    input_shape=input_shape,
                                    output_size=output_dim,
                                    learning_rate=args.learning_rate)
        thread_list = []
        env_list = []

        # Thread agent 생성
        for id in range(args.n_threads):
            logdir = args.logdir

            env = gameEnv(partial=False, size=args.env_size)
            # env = gym.make("PongDeterministic-v0")

            single_agent = Agent(env=env,
                                 session=sess,
                                 coord=coord,
                                 id=id,
                                 global_network=global_network,
                                 input_size=input_size,
                                 input_shape=input_shape,
                                 output_dim=output_dim,
                                 learning_rate=args.learning_rate,
                                 update_ep_size=args.update_ep_size,
                                 logdir=logdir)
            single_agent.max_ep_steps = args.max_ep_steps

            if id == 0:
                single_agent.local.summary_writer.add_graph(sess.graph)

            thread_list.append(single_agent)
            env_list.append(env)

        # 모델 초기화
        init = tf.global_variables_initializer()
        sess.run(init)

        # saver 설정
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
        saver = tf.train.Saver(var_list=var_list)

        # Save 파일 있을 경우 복구
        if tf.train.get_checkpoint_state(checkpoint_dir):
            read_path = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess, read_path)

            Agent.global_episode = sess.run(global_network.global_episodes)
            print("Model restored to global")
        else:
            print("No model is found")

        # 모델 그래프 최종 확정
        tf.get_default_graph().finalize()

        print('\nProgram start')
        print('Learning rate :', args.learning_rate)

        # Thread Start
        for t in thread_list:
            t.start()
            time.sleep(1)

        print("Ctrl + C to close")

        # Save step 설정
        div_num = 1000
        save_idx = int(sess.run(global_network.global_episodes) / div_num)

        while not coord.should_stop():
            current_episode = Agent.global_episode
            sess.run(global_network.update_ep, feed_dict={global_network.input_ep: current_episode})
            temp_idx = int(current_episode / div_num)
            # Global step XX 회마다 모델 저장
            if save_idx != temp_idx:
                save_idx = temp_idx

                saver.save(sess, save_path, global_step=current_episode)
                print('Checkpoint Saved to {}'.format(save_path), 'Episode :', current_episode)

            if current_episode >= args.max_steps:
                print("Closing threads")
                coord.request_stop()
                coord.join(thread_list)

            time.sleep(2)

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
        current_episode = Agent.global_episode
        saver.save(sess, save_path, global_step=current_episode)
        print('Checkpoint Saved to {}'.format(save_path), 'Episode :', current_episode)

        sess.close()


# Test용 method
def main_test():
    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        sess = tf.InteractiveSession()

        checkpoint_dir = args.logdir
        save_path = os.path.join(checkpoint_dir, "model.ckpt")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print("Directory {} was created".format(checkpoint_dir))

        # Global network 생성
        global_network = A3CNetwork(name="global",
                                    input_size=input_size,
                                    input_shape=input_shape,
                                    output_size=output_dim,
                                    learning_rate=args.learning_rate)
        logdir = args.logdir

        env = gameEnv(partial=False, size=args.env_size)
        env.show_game = True
        # env = gym.make("PongDeterministic-v0")

        single_agent = Agent(env=env,
                             session=sess,
                             coord=None,
                             id=0,
                             global_network=global_network,
                             input_size=input_size,
                             input_shape=input_shape,
                             output_dim=output_dim,
                             learning_rate=args.learning_rate,
                             update_ep_size=args.update_ep_size,
                             is_training=False,
                             logdir=logdir)

        # 모델 초기화
        init = tf.global_variables_initializer()
        sess.run(init)

        # saver 설정
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
        saver = tf.train.Saver(var_list=var_list)

        # Save 파일 있을 경우 복구
        if tf.train.get_checkpoint_state(checkpoint_dir):
            read_path = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess, read_path)
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