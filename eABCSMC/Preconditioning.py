from scipy import optimize, integrate
import numpy as np
import pandas as pd


def compute_corr(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

class Preconditioner:

    """Empirical ABC-SMC implementation for epidemiological models for data preparation and numerical preconditioning.

    Arguments (also attributes):
        ODEmodel(function) : function defining the differential model (as in Models.py)
        model_type(string) : String

    Attributes:
        true_y(numpy array): compartments' data in a readable form.
        true_yy(numpy array): compartments' data in the format required for eABCSMC and the numerical fit.
        extended_y(numpy array): extended compartments' data in the format required for EABCSMC and the numerical fit.
        t_grid(numpy array): considered time grid.
        Data(pandas dataframe): final dataframe with the compartments' data.
        sigma(numpy array): relative standard deviation on the observations (same format as true_yy).


    The main methods are preprocess_data and fit, needed to initialize eABCSMC.
    In this moment, SIR is not tested, thus it may present some bugs.

    """

    def __init__(self, ODEmodel, model_type='SIRD'):
        self.ODEmodel = ODEmodel
        if model_type == 'SIR' or model_type == 'SIRD' or model_type == 'SEIRD':
            self.model_type = model_type
        else:
            "Model not supported, the class is not guaranteed to work fine."
        self.true_y = None
        self.true_yy = None
        self.extended_y = None
        self.t_grid = None
        self.Data = None
        self.sigma = None

    def preprocess_data(self, data, start=0, eff=30, ext=150, return_data=False):
        """ Data preprocessor to be called in order to fit the preconditioner
        with the data structured in the proper form, assuming data is of the
        same form as obtained from the Datagetter.
        The start parameter signals the number of skipped initial days.
        The eff parameter is the number of days to utilize for fitting the model.
        The ext parameter is the number of days to keep in memory to assess
        the model on unknown data.
        """

        data = data[start:ext]
        ext = data.shape[0]
        data.reset_index(inplace=True, drop=True)
        if self.model_type == 'SIRD' or self.model_type == 'SEIRD':
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

        if self.model_type == 'SIRD' or self.model_type == 'SEIRD':
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

        self.true_y = true_y
        self.extended_y = extended_y
        self.t_grid = t_grid
        self.data = data

        if return_data:
            return true_y, extended_y, t_grid

    def fit_odeint(self, x, *args):
        if self.model_type == 'SIRD':
            fit = integrate.odeint(self.ODEmodel, self.true_y[0], x, args=args)
            fit_p = np.append(fit[:, 0], fit[:, 1])
            return np.append(fit_p, fit[:, 2])

        elif self.model_type == 'SIR':
            fit = integrate.odeint(self.ODEmodel, self.true_y[0], x, args=args)
            return np.append(fit_p, fit[:, 1])

        elif self.model_type == 'SEIRD':
            fit = integrate.odeint(self.ODEmodel, np.concatenate((self.true_y[0], np.zeros(1))), x, args=args)
            fit_p = np.append(fit[:, 0], fit[:, 1])
            return np.append(fit_p, fit[:, 2])


    def fit(self, p0, bounds=None, std = None ):
        """ Fits the preconditioning model. P0 are the starting points for the parameters
        to be used by the numerical solver, bounds are optional bounds for the
        parameters, std is to be used in case of data with different expected
        noise for different days/compartments."""

        true_yy = np.append(self.true_y[:, 0], self.true_y[:, 1])
        true_yy = np.append(true_yy, self.true_y[:, 2])

        self.true_yy = true_yy

        if bounds is None and std is None:
            popt, pcov = optimize.curve_fit(self.fit_odeint, xdata=self.t_grid, ydata=true_yy, p0 = p0, method='trf')
        elif bounds is not None and std is None:
            popt, pcov = optimize.curve_fit(self.fit_odeint, xdata=self.t_grid, ydata=true_yy, p0=p0,
                                        bounds=bounds, method='trf')
        elif std is not None and bounds is None:
            self.sigma = std
            popt, pcov = optimize.curve_fit(self.fit_odeint, xdata=self.t_grid, ydata=true_yy, p0=p0,
                                        bounds=bounds,sigma = self.sigma, method='trf')
        else:
            self.sigma = std
            popt, pcov = optimize.curve_fit(self.fit_odeint, xdata=self.t_grid, ydata=true_yy, p0=p0,
                                        bounds=bounds,sigma = self.sigma, method='trf')


        return popt, pcov
