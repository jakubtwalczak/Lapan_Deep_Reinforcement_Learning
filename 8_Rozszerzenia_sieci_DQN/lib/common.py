import numpy as np
import torch
import torch.nn as nn
import warnings
from datetime import timedelta, datetime
from types import SimpleNamespace
from typing import Iterable, Tuple, List

import ptan
import ptan.ignite as ptan_ignite
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger

# jest to wspólna biblioteka dla sieci DQN zaimplementowanej wysokopoziomowo

SEED = 123 # ziarno losowe

HYPERPARAMS = { # słowniki parametrów
    'pong': SimpleNamespace(**{ # klasa będąca ogólnym kontenerem dla różnych wartości
        'env_name':         "PongNoFrameskip-v4", # nazwa środowiska
        'stop_reward':      18.0, # nagroda, której osiągnięcie zatrzymuje trening
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0, # epsilon początkowy
        'epsilon_final':    0.02, # epsilon docelowy
        'learning_rate':    0.0001, # współczynnik uczenia
        'gamma':            0.99,
        'batch_size':       32
    }),
    'breakout-small': SimpleNamespace(**{
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout-small',
        'replay_size':      3*10 ** 5,
        'replay_initial':   20000,
        'target_net_sync':  1000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       64
    }),
    'breakout': SimpleNamespace(**{
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout',
        'replay_size':      10 ** 6,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32
    }),
    'invaders': SimpleNamespace(**{
        'env_name': "SpaceInvadersNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'breakout',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    }),
}


def unpack_batch(batch: List[ptan.experience.ExperienceFirstLast]): # funkcja pobierająca paczkę przejść
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    # state - obserwacja ze środowiska
    # actions - akcja wykonana przez agenta (w postaci l. całkowitej)
    # reward - nagroda zdyskontowana (jeżeli step_count > 1, w przeciwnym razie natychmiastowa)
    # last_state - ostatnia obserwacja z łańcucha doświadczeń (lub wartość None, jeżeli ostatni krok w środowisku)
    for exp in batch:
        state = np.array(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = state
            # tablica przechowuje stan początkowy dla przejść końcowych
            # wynik i tak zostanie zamaskowany
        else:
            lstate = np.array(exp.last_state)
        last_states.append(lstate)
    # paczka przejść jest konwertowana na tablice NumPy
    return np.array(states, copy=False), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"): # funkcja straty
    states, actions, rewards, dones, next_states = \
        unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad(): # zapobieganie wyznaczaniu gradientów
        next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[done_mask] = 0.0

    bellman_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, bellman_vals)


class EpsilonTracker: # klasa implementująca zmniejszanie wartości epsilon podczas treningu
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector,
                 params: SimpleNamespace):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - \
              frame_idx / self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final, eps)


def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer,
                    initial: int, batch_size: int): # klasa pobierająca paczki treningowe z bufora
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)


@torch.no_grad()
def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


def setup_ignite(engine: Engine, params: SimpleNamespace, # funkcja dołączająca procedury obsługi
                 # zgodne z biblioteką Ignite
                 exp_source, run_name: str,
                 extra_metrics: Iterable[str] = ()):
    # pozbycie się ostrzeżenia o brakującym wskaźniku
    warnings.simplefilter("ignore", category=UserWarning)

    handler = ptan_ignite.EndOfEpisodeHandler(
        exp_source, bound_avg_reward=params.stop_reward) # procedura generująca zdarzenie biblioteki Ignite
        # gdy kończy się epizod gry
    handler.attach(engine)
    ptan_ignite.EpisodeFPSHandler().attach(engine) # monitor czasu trwania epizodu i liczby interakcji
    # wyznacza na tej podstawie liczbę FPS

# funkcje obsługi zdarzeń

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine): # wywoływana po zakończeniu epizodu
        passed = trainer.state.metrics.get('time_passed', 0)
        print("Epizod %d: nagroda=%.0f, kroki=%s, "
              "prędkość=%.1f f/s, upłynęło=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps,
            trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=int(passed)))) # wyświetla informacje o epizodzie w konsoli

    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine): # wywoływana w momencie osiągnięcia nagrody granicznej
        passed = trainer.state.metrics['time_passed']
        print("Gra ukończona w ciągu %s sekund, po %d epizodach "
              "i %d iteracjach!" % (
            timedelta(seconds=int(passed)),
            trainer.state.episode, trainer.state.iteration)) # wyświetla komunikat o zakończeniu gry
        trainer.should_terminate = True # zatrzymuje proces treningu

# dalsza część dotyczy danych wyświetlanych w Tensorboard

    now = datetime.now().isoformat(timespec='minutes')
    logdir = f"runs/{now}-{params.run_name}-{run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir) # przesyłanie danych do TensorBoard
    run_avg = RunningAverage(output_transform=lambda v: v['loss']) # wygładzona wartość straty
    run_avg.attach(engine, "avg_loss")

    metrics = ['reward', 'steps', 'avg_reward'] # wskaźniki obliczane podczas treningu
    handler = tb_logger.OutputHandler(
        tag="episodes", metric_names=metrics)
    event = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # co 100 iteracji dane wysyłane do TensorBoard
    ptan_ignite.PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps'] # wskaźniki dot. procesu trenowania
    metrics.extend(extra_metrics)
    handler = tb_logger.OutputHandler(
        tag="train", metric_names=metrics,
        output_transform=lambda a: a)
    event = ptan_ignite.PeriodEvents.ITERS_100_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)
