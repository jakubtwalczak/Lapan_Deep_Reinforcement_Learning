import torch
import torch.nn as nn

# Bloki konstrukcyjne sieci neuronowej

l = nn.Linear(2, 5) # 2 wejścia, 5 wyjść
# klasa Linear implementuje warstwę ze sprzężeniem wyprzedzającym i opcjonalnym przesunięciem
v = torch.FloatTensor([1, 2])
print(l(v))

# wszystkie klasy pakietu torch.nn dziedziczą po klasie nn.Module
# klasy też można użyć do implementacji wysokopoziomowych bloków sieci neuronowej
# metody klas potomnych dla nn.Module:
# parameters() - zwraca iterator dla wszystkich zmiennych wymagających obliczenia gradientu
# zero_grad() - inicjalizuje wszystkie gradienty parametrów wartościami zerowymi
# to(device) - przenosi wszystkie parametry modułu do danego urządzenia (CPU lub GPU)
# state_dict() - zwraca słownik ze wszystkimi parametrami modułu; przydatna podczas jego serializacji
# load_state_dict() - inicjalizuje moduł przy użyciu słownika stanów

# klasa Sequential pozwala łączyć warstwy i tworzyć potok

s = nn.Sequential(
    nn.Linear(2, 5),
    nn.ReLU(),
    nn.Linear(5, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.Dropout(p=0.3),
    nn.Softmax(dim=1))

print(s)

# ww. model - trójwarstwowa sieć
# pierwszy wymiar danych wyjściowych - funkcja softmax (zerowy wymiar - próbki z paczek)
# funkcja aktywacji - ReLU
# warstwa Dropout na poziomie 30%

print(s(torch.FloatTensor([[1, 2]]))) # zastosowanie modelu na przykładowych danych

# Warstwy definiowane przez użytkownika

# dziedziczenie po klasie nn.Module zapewnia bogate funkcjonalności:
# monitorowanie elementów składowych bieżącego modułu
# wykonywanie funkcji służących do zarządzania parametrami modułów składowych
# klasa ta definiuje konwencję stosowania klas potomnych z danymi
# funkcje dodatkowe, jak rejestrowanie funkcji przechwytującej
# (np. do modyfikacji transformacji lub przepływu gradientów)

# modele składowe można zagnieżdżać, tworząc struktury wyższego poziomu
# by utworzyć moduł niestandardowy, wystarczy wykonać dwie czynności:
# zarejestrować moduły podrzędne
# zaimplementować metodę forward()

class OurModule(nn.Module): # dziedziczenie po nn.Module
    def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
        # trzy parametry: rozmiar wejścia, rozmiar wyjścia i prawdopodobieństwo dropoutu
        super(OurModule, self).__init__()
        self.pipe = nn.Sequential( # przypisanie sieci Sequential do pola "pipe"
            # dzięki temu automatycznie rejestrujemy moduł
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1)
        )

    def forward(self, x): # zastąpienie oryginalnej metody forward()
        # jest wywoływana dla każdej paczki
        return self.pipe(x)

# ogólny schemat pętli treningowej w pseudokodzie

'''
for batch_x, batch_y in iterate_batches(data, batch_size=32): 
    batch_x_t = torch.tensor(batch_x) # przetworzenie próbek danych na tensor
    batch_y_t = torch.tensor(batch_y) # przetworzenie etykiet na tensor
    out_t = net(batch_x_t) # przekazanie próbek do sieci
    loss_t = loss_function(out_t, batch_y_t) # dostarczenie wyników i etykiet celem wyliczenia funkcji straty
    loss_t.backward() # obliczenie wartości gradientów przy pomocy f. straty dla całej sieci
    optimizer.step() # optymalizator pobiera i stosuje wszystkie gradienty
    optimizer.zero_grad() # wyzerowanie gradientów
    '''

if __name__ == '__main__':
    net = OurModule(num_inputs=2, num_classes=3) # żądana liczba wejść i wyjść
    v = torch.FloatTensor([[2, 3]]) # definiujemy tensor
    out = net(v)
    print(net) # informacja o wewnętrznej strukurze sieci
    print(out) # wynik transformacji sieci
