# definiowanie środowiska

import random

class Environment: # inicjalizacja stanu wewnętrznego środowiska
    def __init__(self):
        self.steps_left = 10 # liczba kroków możliwych do wykonania przez agenta

    def get_observation(self): # metoda zwracająca obserwację bieżącego środowiska do agenta
        return [0.0, 0.0, 0.0] # wektor obserwacji wynosi 0 - środowisko nie ma w zasadzie stanu wewnętrznego

    def get_actions(self) -> list[int]: # lista możliwych do wykonania akcji
        return [0, 1] # w tym przykładzie - tylko dwie możliwe akcje

    def is_done(self) -> bool: # informuje agenta o zakończeniu epizodu
        return self.steps_left == 0
        # epizody mogą być skończone lub nieskończone
        # środowisko powinno zapewnić sposób wykrycia, że epizod się zakończył

    def action(self, action: int) -> float: # centralny składnik środowiska
        # wykonuje 2 działania: obsłużenie akcji lub zwrócenie nagrody
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1 # akcja w tym przykładzie jest odrzucana
        return random.random() # nagroda jest losowa

# klasa zarządzająca agentem

class Agent:
    def __init__(self):
        self.total_reward = 0.0 # sumaryczna nagroda zebrana podczas epizodu

    def step(self, env: Environment):
        current_obs = env.get_observation() # obserwacja bieżącego środowiska
        actions = env.get_actions() # lista możliwych do wykonania akcji
        reward = env.action(random.choice(actions)) # wykonanie losowej akcji
        self.total_reward += reward # zdobycie nagrody za wykonanie akcji

if __name__ == '__main__': # kod tworzący obie klasy i uruchamiający jeden epizod
    env = Environment()
    agent = Agent()
    while not env.is_done():
        agent.step(env)
    print(f"Total reward {agent.total_reward:.4f}")