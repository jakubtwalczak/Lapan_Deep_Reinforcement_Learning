import gymnasium as gym
from lib import dqn_model
from lib import wrappers

from dataclasses import dataclass
import argparse
import time
import numpy as np
import collections
import typing as tt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard.writer import SummaryWriter

# Parametry środowiska i treningu
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"  # środowisko
MEAN_REWARD_BOUND = 19  # próg nagrody dla ostatnich 100 epizodów, po przekroczeniu której kończy się uczenie

# Hiperparametry DQN
GAMMA = 0.99  # wartość gamma w przybliżeniu Bellmana
BATCH_SIZE = 32  # rozmiar paczki pobieranej z bufora
REPLAY_SIZE = 10000  # maksymalna pojemność bufora
LEARNING_RATE = 1e-4  # współczynnik uczenia
SYNC_TARGET_FRAMES = 1000  # częstotliwość synchronizacji wag modelu treningowego z docelowym
# który to model docelowy jest używany do uzyskania wartości następnego stanu w przybliżeniu Bellmana
REPLAY_START_SIZE = 10000  # liczba ramek gromadzona przed rozpoczęciem treningu do zapełnienia bufora

EPSILON_DECAY_LAST_FRAME = 150000  # liczba klatek, po których osiągnięta zostaje minimalna wartość epsilon
EPSILON_START = 1.0  # wartość epsilon na początku treningu (wszystkie akcje wybierane losowo)
EPSILON_FINAL = 0.01  # wartość epsilon na końcu treningu (1% akcji wybierany losowo)

# Typy danych
State = np.ndarray
Action = int
BatchTensors = tt.Tuple[
    torch.ByteTensor,  # aktualny stan
    torch.LongTensor,  # podjęte akcje
    torch.Tensor,  # nagrody
    torch.BoolTensor,  # czy epizod zakończony
    torch.ByteTensor  # nowy stan
]


@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    done_trunc: bool
    new_state: State


# bufor przechowujący przejścia ze środowiska w postaci krotek
# liczba kroków jest stała - 10 tys. ("REPLAY_START_SIZE")
# na potrzeby trenowania pobieramy paczkę przejść z bufora

class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> tt.List[Experience]:
        indices = np.random.choice(len(self), batch_size, replace=False)  # tworzenie listy losowych indeksów
        return [self.buffer[idx] for idx in indices]


# agent współdziała ze środowiskiem i zapisuje wyniki w buforze
# musimy pamiętać referencje do środowiska i bufora podczas inicjalizacji agenta
# dzięki temu będzie mógł mieć dostęp do bieżących obserwacji i nagrody sumarycznej

class Agent:
    def __init__(self, env: gym.Env, exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state: tt.Optional[np.ndarray] = None
        self._reset()

    def _reset(self):
        self.state, _ = env.reset()  # Reset środowiska
        self.total_reward = 0.0

    # główna metoda wykonuje krok w środowisku i zapisuje wyniki w buforze
    # wybór akcji następuje z uwzględnieniem parametru epsilon
    # w przeciwnym razie używa poprzedniego modelu dla uzyskania wartości Q dla wszystkich możliwych akcji i wyboru najlepszej z nich

    @torch.no_grad()
    def play_step(self, net: dqn_model.DQN, device: torch.device,
                  epsilon: float = 0.0) -> tt.Optional[float]:
        done_reward = None

        # Wybór akcji zgodnie z polityką epsilon-greedy
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Losowa akcja
        else:
            state_v = torch.as_tensor(self.state).to(device)
            state_v.unsqueeze_(0)  # Dodanie wymiaru batcha
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # Wykonanie akcji w środowisku
        new_state, reward, is_done, is_tr, _ = self.env.step(action)  # wykonanie kroku
        self.total_reward += reward  # sumowanie nagrody

        # Zapis doświadczenia do bufora
        exp = Experience(
            state=self.state, action=action, reward=float(reward),
            done_trunc=is_done or is_tr, new_state=new_state
        )
        self.exp_buffer.append(exp)  # zapis w buforze
        self.state = new_state  # uzyskanie nowego stanu

        # Reset stanu po zakończeniu epizodu
        if is_done or is_tr:
            done_reward = self.total_reward
            self._reset()
        return done_reward  # zwrócenie nagrody sumarycznej


# Przekształcenie batcha doświadczeń na tensory PyTorch
def batch_to_tensors(batch: tt.List[Experience], device: torch.device) -> BatchTensors:
    states, actions, rewards, dones, new_state = [], [], [], [], []  # utworzenie pustych tensorów
    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done_trunc)
        new_state.append(e.new_state)  # przekazanie pozycji z paczki do tensotów
    states_t = torch.as_tensor(np.asarray(states))
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards)
    dones_t = torch.BoolTensor(dones)
    new_states_t = torch.as_tensor(np.asarray(new_state))
    return states_t.to(device), actions_t.to(device), rewards_t.to(device), \
           dones_t.to(device), new_states_t.to(device)  # przekazanie tensorów do CPU/GPU


# Funkcja obliczająca stratę (loss) dla DQN
def calc_loss(batch: tt.List[Experience], net: dqn_model.DQN, tgt_net: dqn_model.DQN,
              device: torch.device) -> torch.Tensor:
    # przekazujemy funkcji jako argumenty:
    # paczkę jako krotkę tablic umieszczonych w buforze, sieć trenowaną i sieć docelową
    # model "net" używamy do wyznaczenia gradientów
    # model "tgt_net" używamy do obliczania wartości dla kolejnych stanów
    # obliczenia te nie powinny wpływać na gradienty - metodą detach() zapobiegamy ich propagacji w sieci docelowej
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(batch, device)
    # tablice NumPy razem z paczkami zamieniami na tensory i kopiujemy do CPU/GPU (w zależności od wyboru)
    state_action_values = net(states_t).gather(
        1, actions_t.unsqueeze(-1)
    ).squeeze(-1)  # przekazanie obserwacji do 1. modelu
    # + operacją gather() wyodrębniamy wartości Q dla podjętych akcji
    # dla tej metody 1. argument to indeks wymiary, dla którego wykonać chcemy operację
    # argument 2. to tensor z indeksami wybranych elementów
    # wywołania squeeze() i unsqueeze() są wymagane do wyznaczenia poprawnego argumentu indeksu dla gather()
    # i pozbycia się dodatkowych wymiarów

    with torch.no_grad():
        next_state_values = tgt_net(new_states_t).max(1)[0]  # obliczenie max. wartości Q dla akcji
        next_state_values[dones_t] = 0.0  # jeśli przejście w paczce pochodzi z ostatniego kroku epizodu
        # wartość akcji nie uwzględnia zdyskontowanej nagrody za stan następny, ponieważ już go nie ma
        next_state_values = next_state_values.detach()  # oddzielenie wartości od wykresu obliczeniowego
        # w celu zapobieżenia przepływowi gradientów do sieci używanej do wyznaczenia przybliżenia Q dla następnych stanów
        # metoda detach() zwraca tensor bez uwzględnienia historii jego obliczeń

    expected_state_action_values = next_state_values * GAMMA + rewards_t  # wartość śr. przybliżenia Bellmana
    return nn.MSELoss()(state_action_values, expected_state_action_values)  # wartość straty


# Główna pętla treningowa
if __name__ == "__main__":  # pętla treningowa
    parser = argparse.ArgumentParser()  # parser argumentów wiersza poleceń
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Użyj technologii CUDA")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Nazwa środowiska. Wartość domyślna=" +
                             DEFAULT_ENV_NAME)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)  # utworzenie środowiska

    net = dqn_model.DQN(env.observation_space.shape,  # utworzenie sieci treningowej
                        env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape,  # utworzenie sieci docelowej
                            env.action_space.n).to(device)
    # obie utworzone sieci są synchronizowane co 1000 klatek (jeden epizod gry) i mają pierwotnie losowe wagi
    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)  # utworzenie buforu
    agent = Agent(env, buffer)  # przekazanie buforu agentowi
    epsilon = EPSILON_START  # inicjalizacja epsilonu

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)  # optymalizator
    total_rewards = []  # bufor nagród za pełne epizody
    frame_idx = 0  # licznik ramek
    ts_frame = 0
    ts = time.time()
    best_m_reward = None  # zmienne do monitoringu prędkości działania programu i najlepszej śr. nagrody

    while True:
        frame_idx += 1  # zwiększenie liczby wykonanych iteracji
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      frame_idx / EPSILON_DECAY_LAST_FRAME)  # zmniejszanie wartości epsilona wg przyjętej metody

        reward = agent.play_step(net, epsilon, device=device)  # wykonanie pojedynczego kroku w środowisku
        # funkcja zwraca wynik None, gdy krok jest ostatni w epizodzie
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)  # obliczenie szybkości (fps)
            ts_frame = frame_idx  # liczba epizodów wykonanych
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])  # śr. nagroda za ostatnie 100 epizodów
            print("%d: gry - %d, nagroda %.3f, "
                  "eps %.2f, %.2f fps" % (
                      frame_idx, len(total_rewards), m_reward, epsilon,
                      speed  # zwracamy też aktualny epsilon
                  ))
            # przekazujemy wszystkie te informacje do TensorBoard
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:  # gdy śr. nagroda za ostatnie 100 epizodów osiąga max.
                torch.save(net.state_dict(), args.env +
                           "-best_%.0f.dat" % m_reward)  # zapisujemy parametry modelu
                if best_m_reward is not None:
                    print("Nagroda uległa zmianie: %.3f -> %.3f" % (
                        best_m_reward, m_reward))  # wyświetlamy komunikat
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:  # jeżeli śr. nagroda osiąga wartość graniczną, kończymy trening
                print("Rozwiązano po %d klatkach!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:  # sprawdzamy, czy bufor jest wystarczająco duży dla procesu trenowania
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:  # synchronizacja parametrów sieci treningowej z docelową co 1000 klatek
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()  # zerowanie gradientów
        batch = buffer.sample(BATCH_SIZE)  # pobranie paczki z bufora
        loss_t = calc_loss(batch, net, tgt_net, device=device)  # kalkulacja straty
        loss_t.backward()
        optimizer.step()  # optymalizacja
    writer.close()
