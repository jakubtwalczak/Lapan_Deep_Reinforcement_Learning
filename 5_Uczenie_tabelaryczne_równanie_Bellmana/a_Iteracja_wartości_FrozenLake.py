import gymnasium as gym
import collections
from tensorboardX import SummaryWriter

# wartości stałe
ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20

class Agent:  # klasa przechowująca tabele i zawierająca funkcje z pętli treningowej
    def __init__(self):
        self.env = gym.make(ENV_NAME)  # środowisko
        self.state = self.reset_env()  # pierwsza obserwacja
        self.rewards = collections.defaultdict(float)  # tabela nagród
        self.transits = collections.defaultdict(collections.Counter)  # tabela przejść
        self.values = collections.defaultdict(float)  # tabela wartości

    def reset_env(self):
        state, _ = self.env.reset()
        return state if isinstance(state, int) else state[0]  # Zapewniamy, że `state` to int

    def play_n_random_steps(self, count):  # funkcja do zbierania losowego doświadczenia ze środowiska
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _, _ = self.env.step(action)
            new_state = new_state if isinstance(new_state, int) else new_state[0]  # Upewniamy się, że `new_state` to int

            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.reset_env() if is_done else new_state # aktualizacja tabel nagród, przejść i stanów

    def calc_action_value(self, state, action): # funkcja obliczająca wartość akcji na podstawie stanu
        # korzysta ona z ww. tabel
        # używa się jej, aby wybrać najlepszą akcję i obliczyć nową wartość stanu podczas iteracji wartości
        target_counts = self.transits[(state, action)] # licznik przejść dla danego stanu i akcji
        total = sum(target_counts.values()) # sumujemy wszystkie liczniki
        action_value = 0.0
        for tgt_state, count in target_counts.items(): # przetworzenie stanu docelowego za pomocą pętli
            reward = self.rewards[(state, action, tgt_state)]
            val = reward + GAMMA * self.values[tgt_state] # udział stanu w całkowitej wartości akcji
            # jest równy natychmiastowej nagrodzie powiększonej o zdyskontowaną wartość dla stanu docelowego
            action_value += (count / total) * val # mnożymy otrzymaną sumę przez prawdopodobieństwo przejścia
            # i dodajemy do końcowej wartości akcji
        return action_value

    def select_action(self, state): # funkcja wykorzystująca poprzednią
        # do podjęcia decyzji o wyborze najlepszej akcji
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action # wygrywa akcja o najwyższej wartości
                # jest to proces deterministyczny; funkcja przeprowadza eksplorację
                # agent postępuje więc zachłannie
        return best_action

    def play_episode(self, env): # funkcja wyznacza najlepszą akcję i wykonuje epizod w środowisku
        total_reward = 0.0
        state, _ = env.reset()  # RESETUJEMY test_env przed rozpoczęciem epizodu
        if isinstance(state, tuple):
            state = state[0]
        while True: # w pętli przetwarzamy stany i sumujemy nagrody
            action = self.select_action(state)
            new_state, reward, is_done, _, _ = env.step(action)
            new_state = new_state if isinstance(new_state, int) else new_state[0]

            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self): # implementacja metody iteracji wartości
        for state in range(self.env.observation_space.n): # przetwarzamy wszystkie stany środowiska
            # obliczamy przy tym wartości dla tych z nich, które są z danego stanu osiągalne
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            self.values[state] = max(state_values) # przypisujemy stanowi maksymalną wartość ze zbioru

if __name__ == '__main__':
    test_env = gym.make(ENV_NAME) # tworzymy środowisko
    agent = Agent() # instancja klasy Agent
    writer = SummaryWriter(comment="-v-iteration") # ustawienie przesyłu danych do Tensorboard
    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100) # wykonujemy 100 losowych kroków dla wypełnienia tabeli nagród
        agent.value_iteration() # następnie iterujemy wartości dla wszystkich stanów
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env) # uruchomienie epizodów testowych
            # przy użyciu tabeli wartości jako polityki
        reward /= TEST_EPISODES # obliczenie średniej nagrody
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Nagroda uległa zmianie: %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80: # przerwanie przy średniej nagrodzie pow. 0,8
            print("Rozwiązano w %d iteracjach" % iter_no)
            break
    writer.close()

# jest to rozwiązanie, które dość szybko, bo już przy 12 do 100 iteracji pozwala odkryć właściwą politykę
# iteracja wartości wykorzystuje poszczególne wartości stanu lub akcji, szacuje prawdopodobieństwo
# i oblicza wartość oczekiwaną
# nie wymaga ona wykonania pełnych epizodów dla rozpoczęcia nauki
