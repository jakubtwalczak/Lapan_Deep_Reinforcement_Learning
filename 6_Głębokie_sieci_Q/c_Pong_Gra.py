import gymnasium as gym
import argparse
import numpy as np
import typing as tt

import torch

from lib import wrappers
from lib import dqn_model

import collections

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Plik modelu") # przekazanie do skryptu nazwy pliku modelu
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Nazwa środowiska, wartość domyślna=" + DEFAULT_ENV_NAME) # przekazanie nazwy środowiska
    parser.add_argument("-r", "--record", required=True, help="Katalog z plikiem wideo") # utworzenie katalogu rejestrującego wideo z gry
    # domyślnie skrypt wyświetla jedynie klatki
    args = parser.parse_args()

    env = wrappers.make_env(args.env, render_mode="rgb_array") # tworzenie środowiska
    env = gym.wrappers.RecordVideo(env, video_folder=args.record)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n) # tworzenie modelu
    state = torch.load(args.model, map_location=lambda stg, _: stg, weights_only=True)
    net.load_state_dict(state) # wczytanie wag z pliku przekazanego w parametrze wiersza poleceń
    # domyślnie klasa torch próbuje wczytać tensory do urządzenia, z którego były odczytane
    # jeżeli kopiujemy model wytrenowany przy użyciu GPU do komputera bez GPU, lokalizacje trzeba odwzorować ponownie
    # tutaj nie korzystamy z GPU - proces jest wystarczająco szybki z wykorzystaniem CPU

    state, _ = env.reset()
    total_reward = 0.0
    c: tt.Dict[int, int] = collections.Counter()

    while True: # pętla podobna do funkcji play_step() z klasy Agent
        # nie używamy jednak metody epsilonu zachłannego
        state_v = torch.tensor(np.expand_dims(state, 0))
        q_vals = net(state_v).data.numpy()[0]
        action = int(np.argmax(q_vals))
        c[action] += 1
        state, reward, is_done, is_trunc, _ = env.step(action) # przekazanie akcji do środowiska
        total_reward += reward # wyznaczenie nagrody sumarycznej
        if is_done or is_trunc: # zatrzymanie pętli po zakończeniu epizodu
            break
    print("Sumaryczna nagroda: %.2f" % total_reward)
    print("Liczba akcji:", c) # wyświetlenie wartości
    env.close()