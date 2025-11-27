import math
import random
import numpy as np
import logging
from scipy.stats import norm
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    filename='data.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
class PortfolioObjective:
    def __init__(self, risk_free_rate=0.0, lambda_risk=0.01, lambda_cost=0.1, cost_rate=0.01):
        self.rf = risk_free_rate
        self.lambda_risk = lambda_risk
        self.lambda_cost = lambda_cost
        self.cost_rate = cost_rate

    def returns(self, w, mu):
        return np.dot(w, mu)

    def risk(self, w, cov):
        return np.sqrt(w @ cov @ w)

    def sharpe(self, w, mu, cov):
        r = self.returns(w, mu)
        sigma = self.risk(w, cov)
        if sigma == 0:
            return 0
        return (r - self.rf) / sigma

    def transaction_cost(self, w, w0):
        return self.cost_rate * np.sum(np.abs(w - w0))

    # Composite objective (can be modified freely)
    def composite(self, w, mu, cov, w0):
        return self.sharpe(w, mu, cov)  # ← using your final objective

class State:
    def __init__(self, expected_returns, cov_matrix, risk_free_rate=0.0, max_w=0.3, k=5):
        self.returns = expected_returns
        self.cov = cov_matrix
        self.n = len(expected_returns)
        self.k = k
        self.max_w = max_w
        self.prob_list = []

        # Objective
        self.obj = PortfolioObjective(risk_free_rate=risk_free_rate)

        # Initial weights
        w = np.ones(self.n)
        self.weights = w / w.sum()
        self.initial_weights = self.weights.copy()

        logging.info(f"Initial Sharpe Ratio: {self.sharpe_ratio(self.weights):.4f}")

    def _enforce_max_w(self, w):
        excess = np.clip(w - self.max_w, 0, None)
        total_excess = excess.sum()

        if total_excess == 0:
            return w

        # First clamp
        w = np.minimum(w, self.max_w)
        room = np.clip(self.max_w - w, 0, None)
        room_sum = room.sum()

        if room_sum > 0:
            w += room * (total_excess / room_sum)
        else:  # emergency normalization
            w[:] = self.max_w
            w /= w.sum()

        return w

    # -------------------- Metrics --------------------
    def sharpe_ratio(self, w):
        return self.obj.sharpe(w, self.returns, self.cov)

    def composite_cost(self, w=None):
        if w is None:
            w = self.weights
        val = -self.obj.composite(w, self.returns, self.cov, self.initial_weights)
        logging.info(f"Composite: {val}")
        return val

    def cost_change(self, new_w):
        return self.composite_cost(new_w) - self.composite_cost()

    def get_neighbour(self, temperature):
        w = self.weights.copy()
        active = np.where(w >= 0.01)[0]

        if len(active) >= 2:
            i, j = np.random.choice(active, 2, replace=False)
            max_transfer = min(w[i], 1 - w[j])

            step_size = 0.02 + 0.15 * min(1.0, temperature / 100)
            delta = np.random.uniform(0, step_size) * max_transfer

            w[i] -= delta
            w[j] += delta

        # Clamp + enforce
        w = np.clip(w, 0, 1)
        w = self._enforce_max_w(w)
        w /= w.sum()
        return w

    def update(self, new_weights):
        self.weights = new_weights

def linear_schedule(temp, c):
    return max(0, temp - c)

def exponential_schedule(temp, c):
    return (1 - c) * temp

def logarithmic_schedule(temp, c, step, initial):
    return initial * c / math.log(step + 2)

class Annealer:
    def __init__(self, state, initial_temp, schedule_type, scheduling_constant):
        self.state = state
        self.temperature = initial_temp
        self.initial_temp = initial_temp
        self.schedule_type = schedule_type
        self.c = scheduling_constant
        self.step = 0

    # Temperature update
    def update_temperature(self):
        if self.schedule_type == "linear":
            self.temperature = linear_schedule(self.temperature, self.c)
        elif self.schedule_type == "exponential":
            self.temperature = exponential_schedule(self.temperature, self.c)
        elif self.schedule_type == "logarithmic":
            self.temperature = logarithmic_schedule(
                self.temperature, self.c, self.step, self.initial_temp
            )

    # Acceptance probability
    def calc_prob(self, dE):
        if self.temperature == 0:
            return 0
        return math.exp(-dE / self.temperature)

    # One SA step
    def anneal_step(self):
        new_w = self.state.get_neighbour(self.temperature)
        dE = self.state.cost_change(new_w)

        if dE <= 0 or random.random() < self.calc_prob(dE):
            self.state.update(new_w)

        self.step += 1
        self.update_temperature()

    # Full annealing run
    def anneal(self, steps=10000, stop_temp=1e-12):
        best_cost = float("inf")
        best_w = self.state.weights.copy()
        cost_history = []

        plt.ion()
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        for _ in range(steps):

            self.anneal_step()
            cost = self.state.composite_cost()
            cost_history.append(cost)

            # Track best
            if cost < best_cost:
                best_cost = cost
                best_w = self.state.weights.copy()

            # Update plot every 100 steps
            if self.step % 100 == 0:
                self.plot_step(axes, cost_history, best_cost)

            if self.temperature <= stop_temp:
                break

        plt.ioff()
        self.state.update(best_w)
        return self.state

    def plot_step(self, axes, cost_history, best_cost):
        ax1, ax2 = axes

        # Weights
        ax1.clear()
        ax1.bar(
            [str(i+1) for i in range(self.state.n)],
            self.state.weights
        )
        ax1.set_title(f"Weights | Step {self.step}, Temp={self.temperature:.4g}")

        # Cost
        ax2.clear()
        ax2.plot(cost_history, color="red")
        ax2.axhline(best_cost, color="green", linestyle="--")
        ax2.set_title("Composite Cost Over Time")

        plt.pause(0.01)


# ================================================================
# 6. MAIN SCRIPT
# ================================================================
# Replace with your actual dataset loader (kept identical)
from dataset import get_data

data = get_data()
expected_returns, cov_matrix = data

state = State(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    risk_free_rate=0.03
)

annealer = Annealer(
    state=state,
    initial_temp=10,
    schedule_type="logarithmic",
    scheduling_constant=0.001
)

final_state = annealer.anneal(steps=10000, stop_temp=1e-12)

print("\n" + "="*50)
print("OPTIMIZATION RESULTS")
print("="*50)
print("Optimized Weights:", final_state.weights)
print("Expected Return:", final_state.obj.returns(final_state.weights, expected_returns))
print("Portfolio Risk:", final_state.obj.risk(final_state.weights, cov_matrix))
print("Sharpe Ratio:", final_state.sharpe_ratio(final_state.weights))
print("="*50)
