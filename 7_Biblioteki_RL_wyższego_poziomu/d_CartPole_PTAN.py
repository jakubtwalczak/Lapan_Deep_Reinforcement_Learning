import gymnasium as gym
from ptan.experience import ExperienceFirstLast, ExperienceSourceFirstLast
import ptan
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import typing as tt


HIDDEN_SIZE = 128 # neurony warstwy ukrytej
BATCH_SIZE = 16 # wielkość paczki
TGT_NET_SYNC = 10 # stała określająca, co ile iteracji synchronizujemy wagi sieci
GAMMA = 0.9 # współczynnik nagrody przyszłej
REPLAY_SIZE = 1000 # rozmiar bufora doświadczeń
LR = 1e-3 # współczynnik uczenia
EPS_DECAY = 0.99 # mnożnik epsilonu po każdym epizodzie

# klasa tworząca sieć
class Net(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())


@torch.no_grad() # zablokowanie przepływu gradientów
def unpack_batch(batch: tt.List[ExperienceFirstLast], net: Net, gamma: float):
    states = []
    actions = []
    rewards = []
    done_masks = []
    last_states = []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        done_masks.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    # konwersja obiektów ExperienceFirstLast na tensory wymagane do treningu sieci Q
    states_v = torch.as_tensor(np.stack(states))
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    last_states_v = torch.as_tensor(np.stack(last_states))
    last_state_q_v = net(last_states_v)
    best_last_q_v = torch.max(last_state_q_v, dim=1)[0]
    best_last_q_v[done_masks] = 0.0
    return states_v, actions_v, best_last_q_v * gamma + rewards_v


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions) # tworzymy prostą sieć neuronową
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.ArgmaxActionSelector()
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1, selector=selector) # zachłanny selektor akcji
    agent = ptan.agent.DQNAgent(net, selector) # agentowi przekazujemy sieć i selektor jako argumenty
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA) # bufor powtórek
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE) # bufor doświadczeń
    optimizer = optim.Adam(net.parameters(), LR)

    step = 0
    episode = 0
    solved = False

    while True:
        step += 1
        buffer.populate(1) # pobranie jednej paczki treningowej z bufora

        for reward, steps in exp_source.pop_rewards_steps(): # metoda zwracająca listę krotek
            # z informacją o epizodach zakończonych od poprzedniego jej wykonania
            episode += 1
            print(f"{step}: Zakończony epizod {episode}, nagroda = {reward:.2f}, "
                  f"epsilon = {selector.epsilon:.2f}")
            solved = reward > 150 # warunek zakończenia procesu
        if solved:
            print("Gratulacje!")
            break
        if len(buffer) < 2 * BATCH_SIZE:
            continue
        batch = buffer.sample(BATCH_SIZE)
        states_v, actions_v, tgt_q_v = unpack_batch(batch, tgt_net.target_model, GAMMA) # wypakowanie paczki tensorów
        optimizer.zero_grad()
        q_v = net(states_v)
        q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        loss_v = F.mse_loss(q_v, tgt_q_v) # obliczanie straty
        loss_v.backward() # propagacja wsteczna
        optimizer.step()
        selector.epsilon *= EPS_DECAY # mnożenie epsilonu o 99%

        if step % TGT_NET_SYNC == 0:
            tgt_net.sync() # synchronizacja sieci źródłowej i docelowej
