import math
import random
import logging
import matplotlib.pyplot as plt
logging.basicConfig(filename="data.log",level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

def f(x):
    return x*x + 2*x + 1
def doIt(x0):
    x=x0
    cool_down = 0.9
    no_of_times = 10000
    T = 1000
    x_arr = []
    y_arr = []
    for i in range(no_of_times):
        new_x = x + random.uniform(-1, 1)
        x_arr.append(i)
        y_arr.append(f(x))
        logging.info(f"Current x: {x}, New x: {new_x}, f(x): {f(x)}, f(new_x): {f(new_x)}, Temperature: {T}")
        delta_E = f(new_x) - f(x)
        if delta_E < 0:
            x = new_x
        else:
            p = math.exp(-delta_E / T)
            if random.random() < p:
                x = new_x
        T *= cool_down
    plt.plot(x_arr, y_arr)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function Optimization Path')
    plt.show()
    return x
print(doIt(0))
