# Advantage Actor Critic

import gymnasium as gym
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Hyperparameters
learning_rate = 0.0005
gamma = 0.99
n_rollout = 20
max_epoch = 2000

# Tensorboard
writer = SummaryWriter(log_dir="/mnt/tf_log/cartpole_a2c")


class ActorCritic(nn.Module):
    def __init__(self, n_space: int, n_action: int):
        super(ActorCritic, self).__init__()
        self.data = []

        n_width = 256

        # Actor network
        self.actor_net = nn.Sequential(
            nn.Linear(n_space, n_width),
            nn.ReLU(),
            nn.Linear(n_width, n_action),
            nn.Softmax(dim=-1),
        )

        # Critic network
        self.critic_net = nn.Sequential(
            nn.Linear(n_space, n_width), nn.ReLU(), nn.Linear(n_width, 1)
        )

        # optimizer
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        # scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epoch, eta_min=1e-6
        )

    def forward(self, state: Tensor):
        prob = self.actor_net(state)
        value = self.critic_net(state)

        return prob, value

    def put_data(self, transition) -> None:
        self.data.append(transition)

    def make_batch(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([int(done)])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = (
            torch.tensor(s_lst, dtype=torch.float).to(DEVICE),
            torch.tensor(a_lst).to(DEVICE),
            torch.tensor(r_lst, dtype=torch.float).to(DEVICE),
            torch.tensor(s_prime_lst, dtype=torch.float).to(DEVICE),
            torch.tensor(done_lst).to(DEVICE),
        )
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def get_returns(
        self, next_states: Tensor, rewards: Tensor, done: Tensor, gamma=0.99
    ) -> Tensor:
        """현재 보상 + 앞으로 얻게될 보상의 합 계산

        Args:
            next_states (Tensor): 다음 상태 리스트
            rewards (Tensor): reward 리스트
            done (Tensor): 종료 여부 리스트
            gamma (float, optional): 앞으로 얻게될 보상에 대한 감쇠 값. Defaults to 0.99.

        Returns:
            Tensor: reward 리스트
        """

        # 현재 Trajectory의 마지막 상태에 대한 값 예측, 에피소드가 끝났다면 다음 상태의 반환 값을 전파하지 않음
        _, v_next = self.forward(next_states[-1])

        R: Tensor = v_next * (1 - done[-1])

        # rewards의 역순부터 반환 값 계산
        batch_return = []

        for idx in reversed(range(len(rewards))):
            reward = rewards[idx]
            R = reward + gamma * R
            batch_return.append(R)

        batch_return.reverse()

        return torch.tensor(batch_return).unsqueeze(1).to(DEVICE)

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()

        # 반환 값 계산
        td_target = self.get_returns(s_prime, r, done)

        probs, values = self.forward(s)
        probs_act = probs.gather(1, a)

        advantages = td_target - values

        loss_actor = -(torch.log(probs_act) * advantages.detach()).mean()
        loss_critic = 0.5 * F.mse_loss(values, td_target.detach())

        loss = loss_actor + loss_critic

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()


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

        prob, _ = model(torch.from_numpy(s).float().to(DEVICE))
        m = Categorical(prob)
        a = m.sample().item()
        s_prime, r, done, truncated, info = env_test.step(a)
        s = s_prime

        if done or truncated:
            break


def main():
    env = gym.make("CartPole-v1")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    model.to(DEVICE)
    summary(model, (4, 4))

    acc_score = 0.0
    print_interval = 20

    for n_epi in range(max_epoch):
        done = False
        s, _ = env.reset()
        ep_score = 0.0
        ep_loss = 0.0

        while not done:
            for t in range(n_rollout):
                prob, _ = model(torch.from_numpy(s).float().to(DEVICE))
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)
                model.put_data((s, a, r, s_prime, done))

                s = s_prime
                ep_score += r

                # Episode 종료
                if done or truncated:
                    done = True
                    break

            loss = model.train_net()
            ep_loss += loss

        acc_score += ep_score

        # Tensorboard에 학습 상태 기록
        writer.add_scalar("Train/reward", ep_score, n_epi)
        writer.add_scalar("Train/loss", ep_loss, n_epi)

        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = acc_score / print_interval
            print(f"# of episode :{n_epi}, avg score : {avg_score:.1f}")
            acc_score = 0.0

            # 평균 점수가 일정 이상이면 종료
            # if avg_score >= 490:
            #     break

    env.close()

    # Log 기록
    writer.flush()

    print("모델 학습 완료!!!")

    # 학습된 모델 실제 실행
    test(model)


if __name__ == "__main__":
    main()
