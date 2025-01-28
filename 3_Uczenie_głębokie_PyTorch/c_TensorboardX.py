import math
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    writer = SummaryWriter() # uruchomienie obiektu zapisującego
    funcs = {'sin': math.sin,
             'cos': math.cos,
             'tan': math.tan} #zdefiniowanie funkcji do wyświetlenia
    for angle in range(-360, 360):
        angle_rad = angle * math.pi / 180 # konwersja stopni na radiany
        for name, fun in funcs.items():
            val = fun(angle_rad) # obliczenie wartości funkcji dla kąta
            writer.add_scalar(name, val, angle) # zapisanie nazwy parametru, wartości i bieżącej iteracji
    writer.close() # zamknięcie obiektu zapisującego
    # dane wyjściowe są zapisywane w katalogu runs