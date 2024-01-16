import gymnasium as gym
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

# PPO ( Proximal Policy Optimization )

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.2
K_epoch = 3
max_epoch = 2000
T_horizon = 20

# Tensorboard
writer = SummaryWriter(log_dir="/mnt/tf_log/cartpole_ppo")


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x: Tensor, softmax_dim=0) -> Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x: Tensor):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition) -> None:
        self.data.append(transition)

    def make_batch(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_lst, dtype=torch.float),
            torch.tensor(prob_a_lst),
        )
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(
                torch.log(pi_a) - torch.log(prob_a)
            )  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(
                self.v(s), td_target.detach()
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def test(model: PPO):
    """학습 완료 후 모델 Test

    Args:
        model (PPO): 학습 완료된 모델
    """
    env_test = gym.make("CartPole-v1", render_mode="human")
    s, _ = env_test.reset()
    done = False

    while not done:
        env_test.render()

        prob = model.pi(torch.from_numpy(s).float())
        m = Categorical(prob)
        a = m.sample().item()
        s_prime, r, done, truncated, info = env_test.step(a)
        s = s_prime

        if done or truncated:
            break


def main():
    env = gym.make("CartPole-v1")
    model = PPO()
    summary(model)

    acc_score = 0.0
    print_interval = 20

    for n_epi in range(max_epoch):
        s, _ = env.reset()
        done = False
        score = 0.0

        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)

                model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), done))
                s = s_prime
                score += r

                if done or truncated:
                    done = True
                    break

            model.train_net()

        acc_score += score

        # Tensorboard에 학습 상태 기록
        writer.add_scalar("Train/reward", score, n_epi)

        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = acc_score / print_interval
            print(f"# of episode :{n_epi}, avg score : {avg_score:.1f}")
            acc_score = 0.0

            # 평균 점수가 일정 이상이면 종료
            if avg_score >= 490:
                break

    env.close()

    # Log 기록
    writer.flush()

    print("모델 학습 완료!!!")

    # 학습된 모델 실제 실행
    test(model)


if __name__ == "__main__":
    main()
