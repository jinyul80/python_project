import gymnasium as gym
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

# Vanilla Actor Critic

# Hyperparameters
learning_rate = 0.0005
gamma = 0.99
n_rollout = 20
max_epoch = 2000

# Tensorboard
writer = SummaryWriter(log_dir="/mnt/tf_log/cartpole_vac")


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []

        n_width = 256

        self.fc1 = nn.Linear(4, n_width)
        self.fc_pi = nn.Linear(n_width, 2)
        self.fc_v = nn.Linear(n_width, 1)
        # optimizer
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        # scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epoch, eta_min=1e-5
        )

    def pi(self, x, softmax_dim=0) -> Tensor:
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

    def make_batch(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst, dtype=torch.float),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_lst, dtype=torch.float),
        )
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(
            self.v(s), td_target.detach()
        )

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return loss.detach().numpy().mean()


def test(model: ActorCritic):
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
    model = ActorCritic()
    summary(model)

    acc_score = 0.0
    print_interval = 20

    for n_epi in range(max_epoch):
        done = False
        s, _ = env.reset()
        score = 0.0

        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)
                model.put_data((s, a, r, s_prime, done))

                s = s_prime
                score += r

                # Episode 종료
                if done or truncated:
                    done = True
                    break

            loss = model.train_net()

        acc_score += score

        # Tensorboard에 학습 상태 기록
        writer.add_scalar("Train/reward", score, n_epi)
        writer.add_scalar("Train/loss", loss, n_epi)

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
