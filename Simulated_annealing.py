import math
import random
from typing import Callable
from heapdict import heapdict

class State():
    def __init__(self):
        self.state = []
        pass

    def cost(self, state) -> float:
        pass

    def get_neighbour(self) -> tuple:
        pass

    def update(self, move : tuple) -> None:
        pass

    def cost_change(self, move : tuple) -> float:
        pass

    def get_all_neighbours(self) -> heapdict:
        pass

    def get_affected_moves(self, move : tuple, queue : heapdict) -> heapdict:
        initial_state = self.state.copy()
        self.update(move)
        del queue
        queue = self.get_all_neighbours()
        self.state = initial_state

        return queue

class Annealer():
    def __init__(self, state : State, initial_temp : float, temperature_schedule : str | Callable[[float, int, float], float], scheduling_constant : float):
        self.state = state
        self.temperature = initial_temp
        self.step = 0
        self.temperature_schedule = temperature_schedule
        self.scheduling_constant = scheduling_constant
        self.optimal_state = self.state.state

    def calc_prob(self, energy_change : float) -> float:
        if self.temperature == 0:
            prob = 0
        else:
            prob = math.exp(- energy_change / self.temperature)
        
        return prob
    
    def linear_schedule(self) -> None:
        if (self.temperature >= self.scheduling_constant):
            self.temperature -= self.scheduling_constant
        else:
            self.temperature = 0
    
    def exponential_schedule(self) -> None:
        self.temperature = (1 - self.scheduling_constant) * self.temperature
    
    def logarithmic_schedule(self):
        self.temperature = self.scheduling_constant / math.log(self.step + 2)
    
    def schedule_step(self) -> None:
        if self.temperature_schedule == 'linear':
            self.linear_schedule()
        elif self.temperature_schedule == 'exponential':
            self.exponential_schedule()
        elif self.temperature_schedule == 'logarithmic':
            self.logarithmic_schedule()
        else:
            self.temperature = self.temperature_schedule(self.temperature, self.step, self.scheduling_constant)
    
    def anneal_step(self) -> bool:
        neighbour_move = self.state.get_neighbour()
        del_E = self.state.cost_change(neighbour_move)
        
        if (del_E <= 0):
            self.state.update(neighbour_move)
        else:
            prob = self.calc_prob(del_E)
            if (random.random() <= prob):
                self.state.update(neighbour_move)
            else:
                del_E = 0
        
        self.step += 1
        self.schedule_step()

        return del_E

    
    def anneal(self, steps = None, stop_temp = None, unchanged_threshold = 100, initial_temp = None, reset_temp = 0.5, n_runs = 10, local_search = False) -> State:
        initial_step = self.step
        cost = self.state.cost(None)
        best_cost = cost
        best_state = self.state.state.copy()

        if initial_temp is not None:
            self.temperature = initial_temp
        elif initial_temp == "logarithmic":
            self.scheduling_constant = initial_temp

        initial_temp = self.temperature / reset_temp

        for i in range(n_runs):
            initial_temp = reset_temp * initial_temp
            self.temperature = initial_temp
            self.step = 0

            unchanged_steps = 0

            while True:
                del_E = self.anneal_step()

                cost += del_E

                if cost < best_cost:
                    best_cost = cost
                    best_state = self.state.state.copy()
                    unchanged_steps = 0
                else:
                    unchanged_steps += 1

                if (steps is not None) and ((self.step - initial_step) >= steps):
                    break
                elif (stop_temp is not None) and (self.temperature <= stop_temp):
                    break
                elif (unchanged_steps >= unchanged_threshold):
                    break

            print(f"Run {i+1} / {n_runs}")
            print(f"Temperature : {initial_temp:0.4f}  Best cost : {best_cost}\n")
            
            self.optimal_state = best_state.copy()
            self.state.state = best_state.copy()
            cost = best_cost

        if local_search:
            best_cost = self.greedy_search()

            print(f"Final local search")
            print(f"Best cost : {best_cost}\n")
        
            self.optimal_state = self.state.state.copy()
        
        return self.state
    
    def greedy_search(self):
        final_cost = self.state.cost(None)
        queue = self.state.get_all_neighbours()

        if len(queue) > 0:
            move, del_E = queue.popitem()

        while (len(queue) > 0 and  del_E < 0):
            queue = self.state.get_affected_moves(move, queue)
            self.state.update(move)
            final_cost += del_E
            
            move, del_E = queue.popitem()
        
        return final_cost
    
    def get_solution(self) -> State:
        return self.state
