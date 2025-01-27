import gymnasium as gym
from typing import TypeVar
import random

Action = TypeVar('Action')

# wrapper (opakowanie) to struktura lub mechanizm wprowadzający dodatkowe funkcjonalności
# dziedziczy on po klasie Env
# najczęściej spotykamy:
# 1. ObservationWrapper - modyfikuje dane obserwacji agenta ze środowiska
# 2. RewardWrapper - modyfikuje nagrody zwracane przez środowisko
# 3. ActionWrapper - modyfikuje sposób przekazywania akcji przez agenta
# 4. EnvironmentWrapper - modyfikuje całe środowisko, a więc wszystko wymienione wyżej
# poniższy wrapper zastępuje z prawdopodobieństwem 0,1 (10%) akcje agenta przypadkowymi
# w ten sposób agent, eksplorując środowisko, od czasu do czasu oddala się od swojej polityki

class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1): # wskazanie prawdopodobieństwa, poniżej którego zastępujemy akcje losowymi
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            print('Randomness!') # przesłaniamy metodę w klasie nadrzędnej - względy abstrakcji
            return self.env.action_space.sample()
        return action


if __name__ == '__main__':
    env = RandomActionWrapper(gym.make('CartPole-v1'))
    obs = env.reset()
    total_reward = 0.0
    while True:
        obs, reward, done, truncated, info = env.step(0) # wykonujemy cały czas akcję 0
        total_reward += reward
        if done:
            break
    print(f"Reward: {total_reward}")