import numpy as np
import matplotlib.pyplot as plt


class Solver:
    def __init__(self, fun, t0, y0, h):
        pass

    def plot_solution(self, t, y, filename):
        plt.plot(t, y)
        plt.xlabel('t')
        plt.ylabel('y')
        plt.savefig(filename)