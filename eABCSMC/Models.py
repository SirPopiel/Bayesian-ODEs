# this file contains many of the models used for fitting. It is, however, easy to define a new one using the same conventions.

import numpy as np
import matplotlib.pyplot as plt

def SIRD(z, t, S0, beta0, omega, gamma, mu):
    I, R, D = z

    S = S0 - I - R - D
    beta = beta0 * np.exp(-omega * t)
    dS = - beta * I * S / S0
    dI = beta * I * S / S0 - gamma * I - mu * I
    dR = gamma * I
    dD = mu * I

    dzdt = [dI, dR, dD]
    return dzdt

# for 30 days fit for italy
def SEIRD(z, t, S0, beta0, gamma, mu0):
    if t == 0:
        I, R, D, E = z
        E = 11.42 * beta0
    else:
        I, R, D, E = z
    S = S0 - I - R - D - E
    mu = mu0 * (1/(t+1))
    alpha = 5.61 * beta0
    beta = beta0 * np.exp(np.exp(-0.0952* beta0 * t) * beta0 * t) 
    dS = - beta * I * S / S0
    dE = beta * I * S / S0 - alpha * E
    dI = alpha * E - gamma * I - mu * I
    dR = gamma * I
    dD = mu * I

    dzdt = [dI, dR, dD, dE]
    return dzdt



# a more complex model,
def SIRD_mu(z, t, S0, beta0, omega, gamma, mu0, mu_const):
    I, R, D = z

    S = S0 - I - R - D
    beta = beta0 * np.exp(-omega * t)
    mu = mu0 * (1 / (t + 1) + mu_const)  # perdita di 0.02 R^2
    dS = - beta * I * S / S0
    dI = beta * I * S / S0 - gamma * I - mu * I
    dR = gamma * I
    dD = mu * I

    dzdt = [dI, dR, dD]
    return dzdt


# 30: [3.57319214e-01, 5.23649174e+05*2, 1.25344524e-02, 5.27022514e-02, 1.99712696e-02, 1.58864240e-02]
# 50: [3.71849011e-01, 4.83512173e+05*2, 1.49740716e-02, 6.18895866e-02, 1.67540531e-02, 1.12222506e-02]
# 60: [3.90754345e-01, 1.00025866e+06*2, 1.71339351e-02, 7.09369720e-02, 1.69477925e-02, 9.41203555e-03]

def SIRD_reduced_60(z, t, beta0, gamma, mu0):
    I, R, D = z

    S0 = 2000517.32 * beta0
    S = S0 - I - R - D
    mu = mu0 * (1 / (t + 1))
    # mu = mu0
    gamma = gamma + np.exp(t - 65) / (1 + np.exp(t - 65)) * (-0.3) * gamma
    beta = beta0 * np.exp(-(1.713e-2 + 7.094e-02 * beta0) * t)
    dS = - beta * I * S / S0
    dI = beta * I * S / S0 - gamma * I - mu * I
    dR = gamma * I
    dD = mu * I

    dzdt = [dI, dR, dD]
    return dzdt


def SIRD_reduced_60_new(z, t, beta0, gamma, kappa, mu0):
    I, R, D = z

    S0 = 2000517.32 * beta0
    S = S0 - I - R - D
    mu = mu0 * (1 / (t + 1))
    gamma = gamma
    beta = beta0 * np.exp(-(1.713e-2 + 7.094e-02 * beta0) * t)
    dS = - beta * I * S / S0
    dI = beta * I * S / S0 - gamma * I - mu * I
    dR = gamma * I
    dD = mu * I

    dzdt = [dI, dR, dD]
    return dzdt


def SIRD_reduced_30(z, t, beta0, gamma, mu0):
    I, R, D = z

    S0 = 1048000 * beta0
    S = S0 - I - R - D
    mu = mu0 * (1 / (t + 1))
    beta = beta0 * np.exp(-(1.2534e-2 + 5.2702e-02 * beta0) * t)
    dS = - beta * I * S / S0
    dI = beta * I * S / S0 - gamma * I - mu * I
    dR = gamma * I
    dD = mu * I

    dzdt = [dI, dR, dD]
    return dzdt


def SIRD_reduced_30_new(z, t, beta0, gamma, kappa, mu0):
    I, R, D = z

    S0 = kappa * 100000 * beta0
    S = S0 - I - R - D
    mu = mu0 * (1 / (t + 1))
    beta = beta0 * np.exp(-(9.72588371e-02 * beta0) * t)
    dS = - beta * I * S / S0
    dI = beta * I * S / S0 - gamma * I - mu * I
    dR = gamma * I
    dD = mu * I

    dzdt = [dI, dR, dD]
    return dzdt


def SIRD_weird(z, t, beta0, alpha0, alpha1, alpha2, alpha3, gamma, mu):
    I, R, D = z

    S0 = (alpha0 + alpha1) * beta0
    S = S0 - I - R - D
    beta = beta0 * np.exp(-(alpha2 + alpha3 * beta0) * t)
    dS = - beta * I * S / S0
    dI = beta * I * S / S0 - gamma * I - mu * I
    dR = gamma * I
    dD = mu * I

    dzdt = [dI, dR, dD]
    return dzdt


def SIR(z, t, S0, beta0, omega, gamma):
    I, R = z
    S = S0 - I - R
    beta = beta0 * np.exp(-omega * t)
    dS = - beta * I * S / N
    dI = beta * I * S / N - gamma * I
    dR = gamma * I

    dzdt = [dI, dR]
    return dzdt


def plot_traj_SIRD(trajectories, width=1.):
    x2 = trajectories[:, 0]
    x3 = trajectories[:, 1]
    x4 = trajectories[:, 2]

    i = plt.plot(x2, linewidth=width, label='Infected')
    r = plt.plot(x3, linewidth=width, label='Recovered')
    d = plt.plot(x4, linewidth=width, label='Deceased')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # plt.title('Real SIRD')
    plt.title('nCov-19 data, Italy')
    plt.tight_layout()


def plot_traj_gray_SIRD(trajectories, width=1.):
    x2 = trajectories[:, 0]
    x3 = trajectories[:, 1]
    x4 = trajectories[:, 2]
    i = plt.plot(x2, linewidth=width, color='lightgray')
    r = plt.plot(x3, linewidth=width, color='lightgray')
    d = plt.plot(x4, linewidth=width, color='lightgray')


def plot_traj_SIRD_united(trajectories, width=1.):
    x2 = trajectories[:, 0]
    x3 = trajectories[:, 1] + trajectories[:, 2]
    i = plt.plot(x2, linewidth=width, label='Infected')
    r = plt.plot(x3, linewidth=width, label='Removed')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title('Real SIRD')


def plot_traj_gray_SIRD_united(trajectories, width=1.):
    x2 = trajectories[:, 0]
    x3 = trajectories[:, 1] + trajectories[:, 2]
    i = plt.plot(x2, linewidth=width, color='lightgray')
    r = plt.plot(x3, linewidth=width, color='lightgray')


def plot_traj_SIR(trajectories, width=1.):
    x2 = trajectories[:, 0]
    x3 = trajectories[:, 1]
    i = plt.plot(x2, linewidth=width, label='Infected')
    r = plt.plot(x3, linewidth=width, label='Removed')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title('Real SIRD')


def plot_traj_gray_SIR(trajectories, width=1.):
    x2 = trajectories[:, 0]
    x3 = trajectories[:, 1]
    i = plt.plot(x2, linewidth=width, color='lightgray')
    r = plt.plot(x3, linewidth=width, color='lightgray')