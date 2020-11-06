import os
import numpy as np
import numpy.random as npr
import tensorflow as tf
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

from scipy.integrate import odeint

keras = tf.keras
tf.compat.v1.enable_eager_execution()

from neural_ode import NeuralODE


if __name__ == "__main__":

    # SIR Model
    def SIR(z, beta, gamma):

        global N
        S, I, R = z

        dS = - beta * I * S / N
        dI = beta * I * S / N - gamma * I
        dR = gamma * I

        dzdt = [dS, dI, dR]
        return dzdt

    def plot_traj(trajectories, width = 1.):
        x1 = trajectories[:,0]
        x2 = trajectories[:,1]
        x3 = trajectories[:,2]
        plt.plot(x1, linewidth = width)
        plt.plot(x2, linewidth = width)
        plt.plot(x3, linewidth = width)

    data_size = 300
    batch_time = 60   # da vedere cosa fa davvero
    niters = 1000
    batch_size = 200  # da vedere cosa fa davvero

    N = 100000
    infected_0 = 50
    beta = 1  # farli time evolving?
    gamma = 0.5

    t_grid = np.linspace(0, data_size, data_size)  # uniformly spaced data? -> even though advantage is learning with not uniformly spaced data
    z0 = [N - infected_0, infected_0, 0] # initial conditions
    true_yy = odeint(SIR, z0, t_grid, args=(beta, gamma))  # potrebbe aver senso tenerli in memoria se è lento