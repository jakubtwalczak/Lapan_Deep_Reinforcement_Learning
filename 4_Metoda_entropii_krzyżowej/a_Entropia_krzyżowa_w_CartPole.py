import gymnasium as gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter
import ale_py

import torch
import torch.nn as nn
import torch.optim as optim

# środowisko CartPole, jak już wspomniałem wcześniej, jest środowiskiem ciągłym
# celem agenta jest niedopuszczenie do upadku drążka przez poruszanie platformą
# za udany ruch otrzymuje się nagrodę 1.0

# zdefiniowane stałe
HIDDEN_SIZE = 128 # liczba warstw ukrytych
BATCH_SIZE = 16 # wielkość paczki
PERCENTILE = 70 # odrzucany % najgorszych nagród


class Net(nn.Module): # definicja klasy
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# obiekty pomocnicze
Episode = namedtuple('Episode', field_names=['reward', 'steps']) # pojedynczy epizod
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action']) # pojedynczy krok


def iterate_batches(env, net, batch_size):
    # funkcja wykorzystująca jako argumenty:
    # 1. środowisko (instancja klasy Env)
    # 2. sieć neuronową (generującą politykę)
    # 3. liczbę epizodów, które należy wygenerować w każdej iteracji
    batch = [] # kontener paczek - epizodów z obiektu Episode
    episode_reward = 0.0 # licznik nagród
    episode_steps = [] # liczba kroków - z obiektu EpisodeStep
    obs, _ = env.reset() # reset środowiska - by uzyskać 1. obserwację
    sm = nn.Softmax(dim=1) # konwersja wyników sieci na rozkład prawdopodobieństwa dla akcji
    # ze względu na to, że dane wyjściowe sieci neuronowej nie były przetwarzane za pomocą funkcji nieliniowej
    while True: # pętla nieskończona
        obs_v = torch.FloatTensor([obs]) # konwersja obserwacji na tensor PyTorch
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        # wyznaczenie rzeczywistej akcji dla bieżącego kroku
        action = np.random.choice(len(act_probs), p=act_probs) # próbkowanie rozkładu prawdopodobieństwa akcji
        next_obs, reward, is_done, _, _ = env.step(action) # przekazanie akcji do środowiska
        episode_reward += reward # dodanie bieżącej nagrody
        step = EpisodeStep(observation=obs, action=action) # zapisanie obserwacji
        # zapisujemy tę obserwację, która służy do wyboru akcji, a nie zwróconą w wyniku wykonania tejże
        episode_steps.append(step)
        # gdy epizod się kończy
        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps) # zapisujemy łączną nagrodę i wykonane kroki w epizodzie
            batch.append(e) # dołączamy epizod do paczki
            episode_reward = 0.0
            episode_steps = [] # resetujemy zmienne nagrody i kroków
            next_obs, _ = env.reset() # resetujemy środowisko
            if len(batch) == batch_size: # sprawdzenie, czy paczka została przetworzona w wymaganej liczbie epizodów
                yield batch # zwrócenie całej paczki epizodów i wstrzymanie dalszego działania funkcji
                batch = [] # wyczyszczenie paczki, aby przygotować ją na nowe epizody
        obs = next_obs # przypisanie obserwacji ze środowiska do zmiennej z bieżącą obserwacją

def filter_batch(batch, percentile): # najważniejsza funkcja w entropii krzyżowej
    # na podstawie paczki epizodów i ułamkowej wartości nagrody wyznacza ograniczenie
    # do filtrowania epizodów trenujących
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile) # odfiltrowanie 30% najlepszych epizodów
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound: # sprawdzenie, czy epizod ma sumaryczną nagrodę większą od ogranicznika
            continue
        train_obs.extend(map(lambda step: step.observation, steps)) # jeżeli tak, wypełniamy listy z obserwacjami...
        train_act.extend(map(lambda step: step.action, steps)) # ...i akcjami wykorzystywanymi do treningu

    # zamiana obserwacji i akcji na tensory
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean # krotka czteroelementowa
    # obserwacja, akcja, ograniczenie nagrody i średnia nagroda


if __name__ == "__main__": # pętla treningowa
    env = gym.make("CartPole-v0") # środowisko
    # env = gym.wrappers.Monitor(env, directory="mon", force=True) # do rejestracji wideo z działań agenta
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions) # sieć neuronowa
    objective = nn.CrossEntropyLoss() # funkcja celu
    optimizer = optim.Adam(params=net.parameters(), lr=0.01) # optymalizator ze współczynnikiem uczenia
    writer = SummaryWriter(comment="-cartpole") # obiekt przesyłający dane do TensorBoard

    for iter_no, batch in enumerate(iterate_batches(
            env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = \
            filter_batch(batch, PERCENTILE) # filtr najlepszych epizodów
        optimizer.zero_grad() # zerowanie gradientów
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v) # funkcja celu
        # oblicza entropię krzyżową dla danych wyjściowych uzyskanych z sieci i działań agenta
        loss_v.backward() # obliczenie gradientu straty
        optimizer.step() # optymalizacja sieci
        # monitorowanie postępu
        print("%d: strata=%.3f, średnia nagroda=%.1f, ograniczenie=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199: # porównanie średnich nagród w epizodach
            # w przypadku CartPole zakłada się, że zadanie kończy się sukcesem
            # jeżeli śr. nagroda za ostatnie 100 epizodów  > 195
            # a długość trwania epizodu jest ograniczona do 200 kroków
            print("Gotowe!")
            break
    writer.close()
