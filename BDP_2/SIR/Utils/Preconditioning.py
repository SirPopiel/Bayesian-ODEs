from scipy import optimize
import numpy as np
import pandas as pd


def preprocess_data(data, start=0, eff=30, ext=150, SIRD=True):
    data = data[start:ext]
    data.reset_index(inplace=True, drop=True)
    if SIRD:
        extended_y = []
        for i in range(ext):
            extended_y.append([data.loc[i, 'Infected'], data.loc[i, 'Recovered'], data.loc[i, 'Dead']])
        extended_y = np.array(extended_y)
    else:
        extended_y = []
        for i in range(ext):
            extended_y.append([data.loc[i, 'Infected'], data.loc[i, 'Recovered'] + data.loc[i, 'Dead']])
        extended_y = np.array(extended_y)

    data = data[:eff]
    data.reset_index(inplace=True, drop=True)

    if SIRD:
        true_y = []
        for i in range(eff):
            true_y.append([data.loc[i, 'Infected'], data.loc[i, 'Recovered'], data.loc[i, 'Dead']])
        true_y = np.array(true_y)
    else:
        true_y = []
        for i in range(eff):
            true_y.append([data.loc[i, 'Infected'], data.loc[i, 'Recovered'] + data.loc[i, 'Dead']])
        true_y = np.array(true_y)

    data['Date'] = pd.to_datetime(data['Date'])

    t_grid = np.arange(eff)

    return true_y, extended_y, t_grid

class Preconditioner:
    def __init__(self, true_y, ODEmodel, num_param):
        self.true_y = true_y
        self.ODEmodel = ODEmodel
        self.num_param = num_param

    def numerical_fit(self):
        popt, pcov = optimize.curve_fit(fit_odeint_reduced_30_new, xdata=t_grid, ydata=true_yy, p0=(0.1, 0.1, 10, 0.1),
                                        bounds=([0, 0, 0, 0], [3, 7, 200, 3]), method='trf')
