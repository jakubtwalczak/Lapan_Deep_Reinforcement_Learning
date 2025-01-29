import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import random
import argparse
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import ale_py
import logging

# sieć GAN to sieć generatywna przeciwstawna
# istnieją w jej ramach dwie sieci
# pierwsza stara się "oszukać" drugą, generując fałszywe próbki danych
# druga - dyskryminator - stara się wykryć sztucznie wygenerowane próbki
# z biegiem czasu obie doskonalą swoje możliwości, zarówno generowania, jak i detekcji
# poniższy kod generuje zrzuty ekranu dla trzech gier z komputera Atari 2600

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16

IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000

class InputWrapper(gym.ObservationWrapper):
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32
        )

    def observation(self, observation):
        new_obs = cv2.resize(
            observation, (IMAGE_SIZE, IMAGE_SIZE))
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)

# klasa opakowuje grę z biblioteki Gymnasium
# zmienia rozdzielczość obrazu wejściowego z 210 X 160 na kwadrat 64 x 64
# przenosi płaszczyznę koloru z ostatniej pozycji do pierwszej
# (żeby spełnić wymagania PyTorch co do warstw konwolucyjnych)
# i rzutuje dane obrazu z bytes na float

class Discriminator(nn.Module): # klasa dyskryminatora dziedziczy po nn.Module
    # wykorzystuje przeskalowany, kolorowy obraz w postaci danych wejściowych
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # potok Sequential poprzez 5 warstw konwolucyjnych zamienia obraz na pojedynczą liczbę
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
            # dane wyjściowe są interpretowane jako prawdopodobieństwo prawdziwości obrazu
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module): # też dziedziczy po nn.Module
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # potok zamienia wektor danych wejściowych na obraz o rozmiarach (3, 64, 64)
        # za pomocą transponowanej konwolucji (dekonwolucji) przekształca go w kolorowy obraz o oryg. rozdzielczości
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)

def iterate_batches(envs, batch_size=BATCH_SIZE):
    # generacja danych wejściowych - przykładowe zrzuty danych
    batch = [e.reset()[0] for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)

    while True: # pętla nieskończona
        e = next(env_gen) # pobranie próbek środowiska z dostępnej listy
        obs, reward, is_done, _, _ = e.step(e.action_space.sample()) # wykonanie losowej akcji
        if np.mean(obs) > 0.01: # dla zapobieżenia migotaniu obrazu w jednej z gier
            batch.append(obs)

        if len(batch) == batch_size: # sprawdzenie, czy paczka osiagnęła żądany rozmiar
            # Debugowanie - sprawdzenie rozmiaru obrazów w batchu
            print("Rozmiary obrazów w batchu:")
            for i, img in enumerate(batch):
                print(f"Obraz {i}: {img.shape}")

            batch_np = np.array(batch, dtype=np.float32)
            batch_np *= 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()

        if is_done:
            e.reset()


if __name__ == '__main__': # przygotowanie modeli i pętli treningowej
    parser = argparse.ArgumentParser()
    parser.add_argument( # przetwarzamy argumenty wiersza poleceń
        "--cuda", default=False, action="store_true",
        help="Włącz opcję CUDA")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")
    envs = [ # pula środowisk przy użyciu zdefiniowanego wrappera
        InputWrapper(gym.make(name))
        for name in ("ALE/Breakout-v5", "ALE/Pong-v5")
    ]
    input_shape = envs[0].observation_space.shape

    net_discr = Discriminator(input_shape=input_shape).to(device)
    net_gener = Generator(output_shape=input_shape).to(device) # obie sieci
    objective = nn.BCELoss() # funkcja straty
    gen_optimizer = torch.optim.Adam(
        params=net_gener.parameters(), lr=LEARNING_RATE,
        betas=(0.5, 0.999))
    dis_optimizer = torch.optim.Adam(
        params=net_discr.parameters(), lr=LEARNING_RATE,
        betas=(0.5, 0.999)) # optymalizatory
    writer = SummaryWriter()
    # w kroku 1., aby wytrenować dyskryminator, dostarczamy mu prawdziwe i fałszywe dane
    # tylko jego parametry aktualizujemy
    # następnie ponownie przetwarzamy za pomocą dyskryminatora próbki
    # tym razem wszystkie próbki mają wartość 1 i aktualizujemy tylko wagi generatora
    # w kroku 2. generator uczy się oszukiwać dyskryminator

    gen_losses = []
    dis_losses = []
    iter_no = 0

    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)
    # definiujemy tablice do gromadzenia wartości strat, liczników iteratorów oraz zmiennych z prawdziwymi i fałszywymi etykietami

    for batch_v in iterate_batches(envs):
        # próbki fikcyjne; dane wejściowe występują w 4 wymiarach: paczka, filtry, x, y
        gen_input_v = torch.FloatTensor(
            BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1).to(device)
        batch_v = batch_v.to(device)
        gen_output_v = net_gener(gen_input_v)
        # tworzymy losowy wektor i przekazujemy go do generatora

        # dyskryminator - trenowany dwa razy
        # najpierw z próbkami prawdziwymi, następnie z wygenerowanymi
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_discr(gen_output_v.detach()) # funkcja detach tworzy niezależną kopię tensora
        # zapobiega ona przenikaniu gradientów do generatora
        dis_loss = objective(dis_output_true_v, true_labels_v) + \
                   objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # generator
        # do dyskryminatora przekazujemy dane z generatora i tym razem nie blokujemy przepływu gradientów
        # stosujemy natomiast funkcję celu z etykietami równymi True
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        # raportujemy wartości strat i przekazujemy próbki obrazów do TensorBoard
        iter_no += 1
        if iter_no % REPORT_EVERY_ITER == 0:
            log.info("Iteracja %d: gen_loss=%.3e, dis_loss=%.3e",
                     iter_no, np.mean(gen_losses),
                     np.mean(dis_losses))
            writer.add_scalar(
                "gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar(
                "dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            writer.add_image("dane fikcyjne", vutils.make_grid(
                gen_output_v.data[:64], normalize=True), iter_no)
            writer.add_image("dane realne", vutils.make_grid(
                batch_v.data[:64], normalize=True), iter_no)

        # proces uczenia jest niezwykle czasochłonny dla komputerów bez GPU
