import gymnasium as gym
import collections
from tensorboardX import SummaryWriter

# wartości stałe
ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
ALPHA = 0.2 # współczynnik uczenia alfa
TEST_EPISODES = 20

# algorytm uczenia Q tabelarycznego przedstawia się następująco:
# 1. rozpoczynamy od pustej tabeli wartości Q(s, a)
# 2. uzyskujemy ze środowiska krotkę (s, a, r, s')
# 3. wykonujemy aktualizację Bellmana
# 4. sprawdzamy warunek konwergencji - jeżeli niespełniony, powtarzamy kroki od 2.

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.reset_env()
        self.values = collections.defaultdict(float)

    def reset_env(self):
        state, _ = self.env.reset()
        return state if isinstance(state, int) else state[0]

    def sample_env(self): # metoda odpowiadająca za wyznaczenie kolejnego przejścia w środowisku
        action = self.env.action_space.sample() # wybór losowej akcji
        old_state = self.state
        new_state, reward, is_done, _, _ = self.env.step(action)
        new_state = new_state if isinstance(new_state, int) else new_state[0]
        self.state = self.reset_env() if is_done else new_state
        return old_state, action, reward, new_state # metoda zwraca krotkę: poprzedni stan, akcja, nagroda, nowy stan

    def best_value_and_action(self, state): # metoda odczytuje stan środowiska i wybiera najlepszą akcję
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value: # wyszukiwanie najwyższej wartości w tabeli
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s): # aktualizacja tabeli wartości
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v # suma nagrody natychmiastowej i zdyskontowanej wartości następnego stanu
        old_v = self.values[(s, a)] # wartości poprzedniego stanu i akcji
        self.values[(s, a)] = old_v * (1-ALPHA) + new_v * ALPHA # łączenie wartości nowych i starych przy użyciu alfa

    def play_episode(self, env): # wykonanie pełnego epizodu
        total_reward = 0.0
        state, _ = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == "__main__": # pętla treningowa
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env() # wykonanie kolejnego kroku
        agent.value_update(s, a, r, next_s) # aktualizacja wartości na podstawie uzyskanych danych

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Nagroda uległa zmianie: %.3f -> %.3f" % (
                best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Rozwiązano w %d iteracjach!" % iter_no)
            break
    writer.close()

# proces treningu jest znacznie dłuższy w porównaniu z metodą iteracji wartości
# nie korzystamy już bowiem z doświadczenia zdobytego podczas testów - nie modyfikowaliśmy statystyk tabeli Q
