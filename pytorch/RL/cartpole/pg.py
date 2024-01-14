import gymnasium as gym
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
max_epoch = 2000


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        # optimizer
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        # scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epoch, eta_min=1e-4
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item) -> None:
        self.data.append(item)

    def train_net(self) -> None:
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


def main() -> None:
    # 환경 생성
    env = gym.make("CartPole-v1")
    # 모델 생성
    pi = Policy()

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
        pi.train_net()

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
            if avg_score >= 490:
                break
    env.close()


if __name__ == "__main__":
    main()
