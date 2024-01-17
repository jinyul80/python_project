import gymnasium as gym
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
max_epoch = 2000

# Tensorboard
writer = SummaryWriter(log_dir="/mnt/tf_log/cartpole_pg")


class Policy(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(Policy, self).__init__()
        self.data = []

        n_width = 16

        self.input_layer = nn.Linear(input_size, n_width)
        self.hidden1 = nn.Linear(n_width, n_width)
        self.hidden2 = nn.Linear(n_width, n_width)
        self.head = nn.Linear(n_width, output_size)
        # optimizer
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        # scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epoch, eta_min=1e-6
        )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.input_layer(x))
        out = F.relu(self.hidden1(out))
        out = F.relu(self.hidden2(out))
        out = F.softmax(self.head(out), dim=0)
        return out

    def put_data(self, item) -> None:
        self.data.append(item)

    def train_net(self):
        # 누적 리워드
        R = 0
        # 모델의 그라디언트 초기화
        self.optimizer.zero_grad()

        # 모델 학습
        # 메모리의 뒤에 있는 샘플부터 순서대로 학습
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            # 모델의 그라디언트 계산(누적)
            loss.backward()

        # 모델의 weight에 그라디언트 적용
        self.optimizer.step()
        # learning rate 축소
        self.scheduler.step()

        # 메모리 초기화
        self.data = []

        return loss.detach().numpy()


def main() -> None:
    # 환경 생성
    env = gym.make("CartPole-v1")
    # 모델 생성
    pi = Policy(env.observation_space.shape[0], env.action_space.n)

    # 모델 레이어 목록 출력
    print(pi)
    print()

    # 변수 설정
    acc_score = 0.0
    print_interval = 20
    score_list, episode_list = [], []

    # 게임 반복 실행
    for n_epi in range(max_epoch):
        s, _ = env.reset()
        score = 0.0

        while True:
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, truncated, info = env.step(a.item())

            pi.put_data((r, prob[a]))
            s = s_prime
            score += r

            # CartPole-v1 forced to terminates at 500 step.
            if done or truncated:
                break

        # 점수 기록
        episode_list.append(n_epi)
        score_list.append(score)
        acc_score += score

        # 모델 학습
        loss = pi.train_net()

        # Tensorboard에 학습 상태 기록
        writer.add_scalar("Train/reward", score, n_epi)
        writer.add_scalar("Train/loss", loss, n_epi)

        if n_epi % print_interval == 0 and n_epi != 0:
            # 학습 결과 그래프로 출력
            plt.figure(figsize=(20, 10))
            plt.plot(episode_list, score_list, "b")
            plt.xlabel("episode", fontsize=20)
            plt.ylabel("score", fontsize=20)
            plt.savefig("./save_graph/cartpole_pg.png")
            plt.close()

            # 평균 점수 출력
            avg_score = acc_score / print_interval
            print(f"# of episode :{n_epi}, avg score : {avg_score}")
            acc_score = 0.0

            # 평균 점수가 일정 이상이면 종료
            # if avg_score >= 490:
            #     break
    env.close()

    # Log 기록
    writer.flush()


if __name__ == "__main__":
    main()
