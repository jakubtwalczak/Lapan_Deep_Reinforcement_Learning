import gymnasium as gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20

# kod dla tej metody jest zbliżony do tego z poprzedniego pliku, z pewnymi modyfikacjami
# tym razem nie zachowujemy wartości stanu, a przechowujemy wartości funkcji Q, która ma stan i akcję jako parametry
# nie potrzebujemy też funkcji "calc_action_value()"

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.reset_env()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def reset_env(self):
        state, _ = self.env.reset()
        return state if isinstance(state, int) else state[0]

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _, _ = self.env.step(action)
            new_state = new_state if isinstance(new_state, int) else new_state[0]
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.reset_env() if is_done else new_state

    def select_action(self, state): # metoda ta przetwarza akcje i wyszukuje ich wartości w tabeli
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)] # metoda wybiera akcję o najwyższej wartości Q
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action # ta akcja jest przyjęta jako wartość stanu docelowego

    def play_episode(self, env):
        total_reward = 0.0
        state, _ = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self): # metoda ta nie opakowuje już funkcji "calc_action_value()"
        # wykonuje za to prawie te same działania
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    key = (state, action, tgt_state)
                    reward = self.rewards[key]
                    best_action = self.select_action(tgt_state)
                    val = reward + GAMMA * \
                          self.values[(tgt_state, best_action)]
                    action_value += (count / total) * val # obliczenie wartości akcji
                    # przy użyciu prawdopodobieństwa stanu docelowego
                self.values[(state, action)] = action_value


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Nagroda uległa zmianie: %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Rozwiązano w %d iteracjach!" % iter_no)
            break
    writer.close()
