import math
import random
import numpy as np
from heapdict import heapdict
import logging
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    filename='data.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
class State:
    def __init__(self):
        self.state = []

    def cost(self, state):
        raise NotImplementedError

    def get_neighbour(self):
        raise NotImplementedError

    def update(self, move):
        raise NotImplementedError

    def cost_change(self, move):
        raise NotImplementedError

    def get_all_neighbours(self):
        raise NotImplementedError

    def get_affected_moves(self, move, queue):
        initial_state = self.state.copy()
        self.update(move)
        del queue
        queue = self.get_all_neighbours()
        self.state = initial_state
        return queue


class Annealer:
    def __init__(self, state, initial_temp,
                 temperature_schedule, scheduling_constant):
        self.state = state
        self.temperature = initial_temp
        self.initial_temp = initial_temp
        self.step = 0
        self.temperature_schedule = temperature_schedule
        self.scheduling_constant = scheduling_constant
        self.optimal_state = self.state.state

    def calc_prob(self, energy_change):
        if self.temperature == 0:
            return 0
        return math.exp(-energy_change / self.temperature)

    def linear_schedule(self):
        self.temperature = max(0, self.temperature - self.scheduling_constant)

    def exponential_schedule(self):
        self.temperature *= (1 - self.scheduling_constant)

    def logarithmic_schedule(self):
        self.temperature = self.initial_temp / math.log(self.step + 2)

    def schedule_step(self):
        if self.temperature_schedule == 'linear':
            self.linear_schedule()
        elif self.temperature_schedule == 'exponential':
            self.exponential_schedule()
        elif self.temperature_schedule == 'logarithmic':
            self.logarithmic_schedule()
        else:
            self.temperature = self.temperature_schedule(
                self.temperature, self.step, self.scheduling_constant
            )

    def anneal_step(self):
        move = self.state.get_neighbour()
        dE = self.state.cost_change(move)
        logging.info(f"Step {self.step}: Move={move}, dE={dE}, Temp={self.temperature}")
        if dE <= 0:
            self.state.update(move)
        else:
            if random.random() <= self.calc_prob(dE):
                self.state.update(move)
            else:
                dE = 0

        self.step += 1
        self.schedule_step()
        return dE

    def anneal(self, steps=None, stop_temp=None,
               unchanged_threshold=100, initial_temp=None,
               reset_temp=0.5, n_runs=10):

        initial_step = self.step
        cost = self.state.cost(None)
        best_cost = cost
        best_state = self.state.state.copy()

        if initial_temp is not None:
            self.temperature = initial_temp

        initial_temp = self.temperature / reset_temp

        for i in range(n_runs):
            initial_temp = reset_temp * initial_temp
            self.temperature = initial_temp
            self.step = 0
            unchanged_steps = 0

            while True:
                dE = self.anneal_step()
                cost += dE

                if cost < best_cost:
                    best_cost = cost
                    best_state = self.state.state.copy()
                    unchanged_steps = 0
                else:
                    unchanged_steps += 1

                if steps is not None and ((self.step - initial_step) >= steps):
                    break
                if stop_temp is not None and self.temperature <= stop_temp:
                    break
                if unchanged_steps >= unchanged_threshold:
                    break

            print(f"Run {i+1}/{n_runs}  Temp={initial_temp:.4f}  Best cost={best_cost}")

            self.optimal_state = best_state.copy()
            self.state.state = best_state.copy()
            cost = best_cost
        best_cost = self.greedy_search()
        self.optimal_state = self.state.state

        print(f"Final local search best cost: {best_cost}")
        return self.state

    def greedy_search(self):
        final_cost = self.state.cost(None)
        queue = self.state.get_all_neighbours()

        move, dE = queue.popitem()

        while dE < 0 and len(queue) > 0:
            queue = self.state.get_affected_moves(move, queue)
            self.state.update(move)
            final_cost += dE
            move, dE = queue.popitem()

        return final_cost

    def get_solution(self):
        return self.state
    def plot_cost_history(self,save=False):
        self.state.plot_cost_history(save)

class PortfolioState(State):
    """
    Move format = (i, j, delta)
      - transfer delta weight from asset i to asset j
    """

    def __init__(self, returns, cov, risk_free_rate=0.0074, max_w=0.3):
        super().__init__()
        self.returns = np.array(returns)
        self.cov = np.array(cov)
        self.rf = risk_free_rate
        self.cost_history=[]
        self.max_w = max_w
        self.n = len(returns)

        # Initial weights
        w = np.ones(self.n)
        self.state = w / w.sum()

        self.initial_state = self.state.copy()

    # Calculating Portfolio Metrics

    def portfolio_return(self, w):
        return np.dot(w, self.returns)

    def portfolio_risk(self, w):
        return np.sqrt(w @ self.cov @ w)

    def sharpe(self, w):
        r = self.portfolio_return(w)
        s = self.portfolio_risk(w)
        return 0 if s == 0 else (r - self.rf) / s

    def cost(self, state):
        # cost = minimize risk
        w = self.state if state is None else state
        return -self.sharpe(w)

    # ---------------------- MOVE MECHANICS ------------------------

    def apply_move(self, w, move):
        """Apply (i, j, delta) without altering self.state."""
        i, j, delta = move
        w2 = w.copy()

        # enforce bounds
        delta = min(delta, w2[i])          
        delta = min(delta, self.max_w - w2[j]) 

        if delta < 0:
            return w.copy()  # no-op safeguard

        w2[i] -= delta
        w2[j] += delta
        return w2

    def get_neighbour(self):
        """Random weight-shift move."""
        i, j = np.random.choice(self.n, 2, replace=False)
        w = self.state

        # max transferable
        max_delta = min(w[i], self.max_w - w[j])
        if max_delta <= 0:
            return self.get_neighbour()  # retry

        delta = np.random.uniform(0, max_delta * 0.2)  # small controlled move

        return (i, j, delta)

    def update(self, move):
        """Update self.state by applying move."""
        self.state = self.apply_move(self.state, move)
        self.cost_history.append(self.cost(self.state))

    def cost_change(self, move):
        w_new = self.apply_move(self.state, move)
        return self.cost(w_new) - self.cost(self.state)

    def get_all_neighbours(self):
        queue = heapdict()
        w = self.state
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue

                max_delta = min(w[i], self.max_w - w[j])
                if max_delta <= 1e-8:
                    continue

                # Discretize — 3 test deltas
                for fraction in (0.05, 0.1, 0.15):
                    delta = max_delta * fraction
                    move = (i, j, delta)
                    dE = self.cost_change(move)
                    queue[move] = dE

        return queue
    def plot_cost_history(self,save=False):
        import matplotlib.pyplot as plt
        plt.plot(self.cost_history)
        plt.xlabel('Steps')
        plt.ylabel('Cost')
        plt.title('Cost History Over Time')
        if save:
            plt.savefig("cost_history.png")
        plt.show()


if __name__ == "__main__":
    from dataset import get_data
    returns, cov = get_data()

    state = PortfolioState(returns, cov)
    annealer = Annealer(
        state=state,
        initial_temp=1,
        temperature_schedule="logarithmic",
        scheduling_constant=0.001
    )

    result = annealer.anneal(
        steps=5000,
        stop_temp=1e-6,
        unchanged_threshold=500,
        n_runs=5
    )
    result.plot_cost_history(save=True)

    print("Optimized weights:", result.state)
    print("Sharpe:", result.sharpe(result.state))
