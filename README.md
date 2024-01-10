# 머신러닝 프레임워크
1. Tensorflow
도커 이미지를 다운 받아서 사용
    ```
    # CPU 버전
    docker pull tensorflow/tensorflow:latest

    # GPU 버전
    docker pull tensorflow/tensorflow:latest-gpu
    ```
    - GPU 사용하는 경우 도커 실행 시 추가 옵션 필요
    "--gpus=all" 옵션 필요
    ```
    docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu \
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```

# Reinforcement Learning(강화학습)을 위한 라이브러리 설치

1. Gym 라이브러리 설치
```
pip install gymnasium "gymnasium[atari]" pygame
```

2. 기타 라이브러리 설치
```
pip install matplotlib pillow
```

3. Gym 라이브러리 실행
```
import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```