import scipy.stats as sps
import numpy.random as npr
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import seaborn as sns
import pandas as pd


class EABCSMCSampler:
    def __init__(self, true_y, ODEmodel, num_param, numerical_estimate, final_time, lambda_ranges = None, prior_means=None, n_jobs=-1, verbose=True):
        self.true_y = true_y
        self.ODEmodel = ODEmodel
        self.num_param = num_param
        self.popt = numerical_estimate
        self.final_time = final_time
        if lambda_ranges is None:
            self.lambda_temp = [npr.uniform(low=0.5, high=5)] * self.num_param
        else:
            self.lambda_temp = lambda_ranges
        if prior_means is None:
            self.prior_means = np.ones(num_param)
        else:
            self.prior_means = prior_means
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.parameters = None
        self.final_weights = None
        self.fitted = False

    def compute_loss(self, params):
        residuals = self.true_y - self.ODEmodel(np.arange(self.final_time), *params)
        loss = np.sum(residuals**2)/1e9
        return loss

    def trGamma_a_inf(self, shape, rate, trunc):
        interval = 1 - sps.gamma.cdf(trunc, a=shape, scale=1/rate)
        yr = npr.rand(1)*interval + sps.gamma.cdf(trunc, a=shape, scale=1/rate)
        xr = sps.gamma.ppf(yr, a=shape, scale=1/rate)
        return xr[0]

    def trGamma_0_a(self, shape, rate, trunc):
        interval = sps.gamma.cdf(trunc, a=shape, scale=1/rate)
        yr = npr.rand(1)*interval
        xr = sps.gamma.ppf(yr, a=shape, scale=1/rate)
        return xr[0]

    def adaptive_gamma_sampling(self, true_center, shape, rate, quantile_1, quantile_2):

        if ((sps.gamma.cdf(true_center, a=shape, scale=1 / rate) < quantile_1) | (
                sps.gamma.cdf(true_center, a=shape, scale=1 / rate) > quantile_2)):
            par = npr.uniform(0, 1)
            loc = sps.gamma.mean(a=shape, scale=1 / rate)
            if true_center > loc:
                trunc = loc + par * (true_center - loc)
                w_temp = self.trGamma_a_inf(shape=shape, rate=rate, trunc=trunc)
            else:
                trunc = loc + par * (true_center - loc)
                w_temp = self.trGamma_0_a(shape=shape, rate=rate, trunc=trunc)
        else:
            w_temp = sps.gamma.rvs(a=shape, scale=1 / rate)

        return w_temp

    def border_estimates_e_abc(self, eps, niters, alpha_quant=0.25):
        # This function estimates the borders of the region containing the eps-approximate posterior through niters
        #  iterations. It uses the empirical sampling method to get nearer to the correct acceptance region.
        #  This is only used to estimate borders, though it can estimate also the parameters in order to avoid
        #  biases in the estimation.

        initial_loss = 0
        loss = []

        def iteration():
            w_temp = [self.adaptive_gamma_sampling(true_center=self.popt[j], shape=self.lambda_temp[j] * self.prior_means[j], rate=self.lambda_temp[j],
                                        quantile_1=alpha_quant, quantile_2=1 - alpha_quant) for j in range(self.num_param)]

            # After the sampling from the prior we go on by simulating the model
            # sim_trajectories = simulate_trajectories_from_theta_hat(model, w_temp)

            sim_loss = self.compute_loss(w_temp)
            loss.append(sim_loss)

            if np.abs(initial_loss - sim_loss) < (eps + npr.uniform() * eps * 1 / 3) and all(w >= 0 for w in w_temp):
                return np.transpose(w_temp)
            return

        if self.verbose:
            parameters = Parallel(n_jobs=self.n_jobs)(delayed(iteration)() for _ in tqdm(range(niters)))
        else:
            parameters = Parallel(n_jobs=self.n_jobs)(delayed(iteration)() for _ in tqdm(range(niters)))

        parameters = np.array(list(filter(None.__ne__, parameters)))

        naccepted = parameters.shape[0]
        print('Acceptance rate: ', naccepted / niters)

        borders = [np.min(parameters[:, i]) for i in range(self.num_param)]

        return borders

    def e_abc_gamma_sampling(self, true_center, shape, rate, border):
        # This function performs the empirical sampling from our method when the "true center" is too far from
        #  the center of the prior.
        """
        loc = sps.gamma.mean(a=shape, scale=1 / rate)
        q = 0.25

        if ((sps.gamma.cdf(true_center, a=shape, scale=1/rate) < q) | (sps.gamma.cdf(true_center, a=shape, scale=1/rate) > 1-q)):
           if true_center < loc:
               w_temp = trGamma_a_inf(shape=shape, rate=rate, trunc=border)
           else:
               w_temp = trGamma_0_a(shape=shape, rate=rate, trunc=border)
        else:
           w_temp = sps.gamma.rvs(a = shape, scale=1/rate)

        """
        w_temp = self.trGamma_a_inf(shape=shape, rate=rate, trunc=border)

        return w_temp

# iperparametro per noise level?
    def preprocessing_e_abc(self, eps, niters, borders, alpha_quant=0.25):
        # This function performs the first sampling from the region defined through the borders estimating function and
        #  returns the estimated parameters and the initializing weights, ([1,1,1,...,1] normalized)

        initial_loss = 0
        loss = []

        def iteration():

            w_temp = [self.e_abc_gamma_sampling(self.popt[j], self.lambda_temp[j] * self.prior_means[j], self.lambda_temp[j],
                                                borders[j]) for j in range(self.num_param)]

            # After the sampling from the prior we go on by simulating the model

            sim_loss = self.compute_loss(w_temp)
            loss.append(sim_loss)

            if np.abs(initial_loss - sim_loss) < (eps + npr.uniform() * eps * 1 / 3) and all(w >= 0 for w in w_temp):
                return np.transpose(w_temp)
            return

        if self.verbose:
            parameters = Parallel(n_jobs=self.n_jobs)(delayed(iteration)() for _ in tqdm(range(niters)))
        else:
            parameters = Parallel(n_jobs=self.n_jobs)(delayed(iteration)() for _ in range(niters))

        parameters = np.array(list(filter(None.__ne__, parameters)))

        naccepted = parameters.shape[0]

        if self.verbose:
            print('Acceptance rate: ', naccepted / niters)

        weights = np.ones(parameters.shape[0]) / parameters.shape[0]

        return parameters, weights

    def compute_weights_abc_smc(self, w, loc, scale, prev_w, prev_p, scale_kernel):
        # This function computes the weights associated to each parameter as described in https://arxiv.org/pdf/1106.6280.pdf

        prob_w = 1
        for i in range(self.num_param):
            prob_w *= sps.norm.pdf(w[i], loc=loc, scale=scale)

        previous_w = 0
        for i in range(prev_w.shape[0]):
            kern_w = 1
            for j in range(self.num_param):
                kern_w *= sps.norm.pdf(w[j], loc=prev_p[i, j], scale=scale_kernel[j])
            previous_w += prev_w[i] * kern_w

        return prob_w / previous_w

    def normalize_weights(self, weights):
        # This function performs the normalization of weights
        tot_weight = np.sum(weights)
        return weights / tot_weight

    def sample_abc_smc_element(self, parameters, weights):
        # This function samples from the previous population according to the specified weights
        elements = np.arange(parameters.shape[0])
        idx = np.random.choice(elements, 1, p=weights)
        return parameters[idx,]

    def perturbation_kernel(self, sdev):
        # This function returns the perturbation from a Gaussian kernel with the specified standard deviation

        return [np.random.randn() * sdev[i] for i in range(self.num_param)]

    def sample_abc_smc(self, eps, niters, kernel_std, old_parameters, weights):
        # This function returns the sampling according to the ABC-SMC with the weights associated to old parameters
        # specified in weights and the old parameters specified in old_parameters

        initial_loss = 0
        loss = []

        def iteration():
            w_temp = self.sample_abc_smc_element(old_parameters, weights)

            # Perturbating with the gaussian Kernel
            pert = self.perturbation_kernel(kernel_std)
            w_temp = w_temp + pert

            w_temp = np.resize(w_temp, (self.num_param,))

            sim_loss = self.compute_loss(w_temp)
            loss.append(sim_loss)

            if np.abs(initial_loss - sim_loss) < (eps + npr.uniform() * eps * 1 / 3) and (min(w_temp) >= 0):
                return np.transpose(w_temp), self.compute_weights_abc_smc(w_temp, 0, 1 / npr.uniform(low=0, high=1.5),
                                                                          weights, old_parameters, kernel_std)
            return None, None

        if self.verbose:
            parameters, new_weights = zip(
                *Parallel(n_jobs=self.n_jobs)(delayed(iteration)(old_parameters, weights, self.num_param) for _ in tqdm(range(niters))))
        else:
            parameters, new_weights = zip(
                *Parallel(n_jobs=self.n_jobs)(delayed(iteration)(old_parameters, weights, self.num_param) for _ in range(niters)))

        parameters = np.array(list(filter(None.__ne__, parameters)))

        new_weights = np.array(list(filter(None.__ne__, new_weights)))

        naccepted = parameters.shape[0]
        print('Acceptance rate: ', naccepted / niters)

        new_weights = self.normalize_weights(new_weights)
        new_weights = new_weights.reshape(naccepted)

        return parameters, new_weights

    def fit(self, epsilon_start=None, niters=3000, eps_schedule=[1, 1/3, 1/5], niters_schedule=[10, 5, 10], return_params = True):

        if epsilon_start is None:
            epsilon_start = 10 * self.compute_loss(self.popt)

        if self.verbose:
            print("Borders estimation start...")
        borders = self.border_estimates_e_abc(epsilon_start * eps_schedule[0], niters * niters_schedule[0])

        if self.verbose:
            print("Borders estimation completed, starting preprocessing...")
            print("Borders:", borders)
        start, start_weights = self.preprocessing_e_abc(epsilon_start * eps_schedule[1], niters * niters_schedule[1], borders)
        if self.verbose:
            print("Preprocessing completed, starting ABC-SMC")

        parameters, weights = self.sample_abc_smc(epsilon_start * eps_schedule[2], niters * niters_schedule[2],
                                                  np.std(start, axis=0), start, start_weights)

        for i in range(2, len(eps_schedule)):
            parameters, weights = self.sample_abc_smc(epsilon_start * eps_schedule[i], niters * niters_schedule[i],
                                                      np.std(parameters, axis=0), parameters, weights)


        self.fitted = True
        self.parameters = parameters
        self.final_weights = weights

        if return_params:
            return parameters

    def pairplot(self):
        if self.fitted:
            g = sns.PairGrid(pd.DataFrame(np.unique(self.parameters, axis=0)))
            g.map_upper(sns.scatterplot, s=5)
            g.map_lower(sns.kdeplot, fill=True)
            g.map_diag(sns.histplot, kde=True)
        else:
            print("Fit model first!")

    #aggiungiamo cose di diagnostica?

