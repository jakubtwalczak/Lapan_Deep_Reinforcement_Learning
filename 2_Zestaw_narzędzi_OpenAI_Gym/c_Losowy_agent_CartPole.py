import gymnasium as gym

if __name__ == '__main__':
    e = gym.make('CartPole-v1') # utworzenie środowiska
    total_reward = 0.0 # inicjalizacja akumulatora nagrody
    total_steps = 0 # inicjalizacja licznika kroków
    obs = e.reset() # reset środowiska
    while True:
        action = e.action_space.sample() # generacja losowej akcji do wykonania
        obs, reward, done, truncated, info = e.step(action) # zwrócenie następnej obserwacji, nagrody i flagi zakończenia
        total_reward += reward # dodanie otrzymanej nagrody
        total_steps += 1 # dodanie kroku
        if done:
            break
    print(f"Episode done in {total_steps} steps, total reward {total_reward}") # wyświetlenie łącznej liczby kroków i nagrody