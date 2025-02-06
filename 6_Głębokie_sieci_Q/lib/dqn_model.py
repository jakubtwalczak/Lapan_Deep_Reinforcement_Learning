import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module): # klasa dziedzicząca po klasie nn.Module
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        # część konwolucyjna
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten() # spłaszczenie tensorów z 4D na 2D
        )

        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]

        # część sekwencyjna
        self.fc = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))


    def forward(self, x: torch.ByteTensor):
        xx = x / 255.0
        return self.fc(self.conv(xx))

# transformacja odbywa się w dwóch krokach:
# 1. stosujemy warstwę konwolucyjną z danymi wejściowymi - na wyjściu otrzymujemy tensor 4D
# wynik zostaje spłaszczony do dwóch wymiarów:
# rozmiar paczki x wektor liczbowy zawierający parametry zwracane przez konwolucję
# ta operacja nie tworzy nowego obiektu w pamięci ani nie przenosi danych w inne miejsce, zmienia tylko kształt tensora
# 2. przekazanie spłaszczonego tensora 2D do warstw w pełni połączonych
