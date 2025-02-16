import gymnasium as gym
import ptan
import argparse
import random
import ale_py

import torch
import torch.optim as optim

from ignite.engine import Engine

from lib import dqn_model, common

NAME = "01_baseline"

if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Użyj technologii CUDA")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params.env_name) # środowisko
    env = ptan.common.wrappers.wrap_dqn(env) # standardowe opakowania z biblioteki wspólnej
    env.seed(common.SEED)

    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device) # instancja modelu sieci

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=params.epsilon_start) # selektor akcji - metoda epsilonu zachłannego
    epsilon_tracker = common.EpsilonTracker(selector, params) # zmniejszanie epsilonu
    agent = ptan.agent.DQNAgent(net, selector, device=device) # instancja agenta

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma) # klasa zwracająca przejścia dla epizodów gry
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size) # klasa tworząca bufor
    optimizer = optim.Adam(net.parameters(),
                           lr=params.learning_rate) # optymalizator

    def process_batch(engine, batch): # funkcja przetwarzania paczek
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(
            batch, net, tgt_net.target_model,
            gamma=params.gamma, device=device) # funkcja straty
        loss_v.backward() # propagacja wsteczna
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME)
    engine.run(common.batch_generator(buffer, params.replay_initial,
                                      params.batch_size))
