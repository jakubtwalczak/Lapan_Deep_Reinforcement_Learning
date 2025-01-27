import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Monitory służą zapisywaniu informacji o działaniach agenta w pliku


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='rgb_array') # tworzymy środowisko z modelem renderowania
    env = RecordVideo(env, "recording", episode_trigger=lambda episode_id: True) # opakowanie środowiska we Wrapper
    # oraz wskazanie nazwy katalogu zapisu
    for episode in range(20): # zapisujemy 20 epizodów w pętli
        obs, info = env.reset() # rozpoczęcie nowego epizodu i rozpoczęcie nowej
        done = False # flaga zakończenia
        while not done:
            action = env.action_space.sample() # wybór losowej akcji
            obs, reward, done, truncated, info = env.step(action) # krotka obserwacji
            env.render() # renderowanie wideo

    env.close() # zamknięcie środowiska