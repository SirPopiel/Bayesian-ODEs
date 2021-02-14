import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import numpy.random as npr
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('agg')

from tensorflow.keras import layers, initializers
from scipy.integrate import odeint


keras = tf.keras
tf.compat.v1.enable_eager_execution()

from neural_ode import NeuralODE  

np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

if __name__ == "__main__":

    # Definition of the Lotka-Volterra model, with VP = f(x(t), t; theta)
    def VP(z, t, alpha, beta, gamma, sigma):
        x, y = z
        dzdt = [alpha * x - beta * x * y, - gamma * y + sigma * x*y]
        return dzdt


    # Function to plot the trajectories in the 2D state space
    def plot_spiral(trajectories, width = 1.):
        x = trajectories[:,0]
        y = trajectories[:,1]
        plt.plot(x, y, linewidth = width)


    # Data of the problem
    data_size = 16001  # Size of the dataset (fictitious and created by adding noise to the real
                       # data found through the integration of the system)
    batch_time = 320  #
    niters = 5000  # Number of Hamiltonian MC iterations
    batch_size = 1000

    # Parameters of the true model
    alpha = 1.
    beta = 0.1
    gamma = 1.5
    sigma = 0.75
        
    t_grid = np.linspace(0, 25, data_size)  # time grid of the same length of data_size
    z0 = [5., 5.]  # Starting point: 5 preys, 5 predators
    true_yy = odeint(VP, z0, t_grid, args=(alpha, beta, gamma, sigma))
    # This calls the Python scipy built-in odeint function in FORTRAN to find the exact values
    # for the trajectories

    true_y = true_yy

    # normalizing data and system
    sigma_x = np.std(true_yy[:,0:1])  # Standard deviation of data for x = preys
    sigma_y = np.std(true_yy[:,1:2])  # Standard deviation of data for y = predators

    noise_level = 0.03  # Adding some noise with this noise level
    sigma_normal = max(sigma_x, sigma_y)  # Considering the highest between the 2

    true_y[:, 0:1] = true_y[:, 0:1]/sigma_x + noise_level * np.random.randn(true_y[:, 0:1].shape[0], true_y[:, 0:1].shape[1])
    true_y[:, 1:2] = true_y[:, 1:2]/sigma_y + noise_level * np.random.randn(true_y[:, 1:2].shape[0], true_y[:, 1:2].shape[1])
    # The 2 lines above normalize the data and then add a noise being extract from a gaussian
    # random variable with 0 mean and (noise_level)^2 variance.


    def get_batch():
        """Returns initial point and last point over sampled frament of trajectory"""
        starts = np.random.choice(np.arange(data_size - batch_time - 1, dtype=np.int64), batch_size, replace=False)
        # This randomly chooses from {0, 1, ... , data_size - batch_time - 1}, batch_size different elements
        batch_y0 = true_y[starts] 
        batch_yN = true_y[starts + batch_time]
        # The function returns a tensor composed by some y0 and the respective yN,
        # being y0 + DeltaT.
        return tf.cast(batch_y0, dtype=tf.float32), tf.cast(batch_yN, dtype=tf.float32)


    time_in = time.time()

    num_param = 4  # Number of parameters
    para_num = num_param

    t0 = t_grid[:batch_time][0]  # t0 = first element of t_grid
    t1 = t_grid[:batch_time][-1]  # t1 = the element of t_grid at batch_time
    t_in = np.linspace(t0, t1, 20)  # The time grid between t0 and t1

    batch_y0, batch_yN = get_batch()  # Returns the first and the last y observed for each batch

    #########################################
    #         precondition start            #
    #########################################

    niters_pre = 500  # Number of iterations of the preconditioner

    class ODEModel_pre(tf.keras.Model):
        def __init__(self):
            super(ODEModel_pre, self).__init__()
            self.Weights = tf.Variable(tf.random.normal([num_param, 1], dtype=tf.float32)*0.01, dtype=tf.float32)
        # Initializer: assign normally distributed random weights which are very close to zero

        def call(self, inputs, **kwargs):
            t, y = inputs
            h = y
            h1 = h[:, 0:1]  # preys
            h2 = h[:, 1:2]  # predators

            p1 = self.Weights[0]
            p2 = self.Weights[1]
            p3 = self.Weights[2]
            p4 = self.Weights[3]

            h_out1 = p1 * h1 + sigma_y * p2 * h2*h1  # Why sigma_y?? Think it's due to normalization, but boh
            h_out2 = p3 * h2 + sigma_x * p4 * h2*h1  # Why sigma_x??
            h_out = tf.concat([h_out1, h_out2], 1)
            # This function is computing the f(x(t), t; p) at [x,t] in 'inputs' and with p
            # the actual weights of the model
            return h_out


    model_pre = ODEModel_pre()  # Here we initialize the model of the preconditioner
    neural_ode_pre = NeuralODE(model_pre, t_in)  # We pass to NeuralODE the actual model
    # and the times we are considering for the step [t0, t1]
    optimizer = tf.compat.v1.train.AdamOptimizer(3e-2)  # The optimizer we're going to use


    @tf.function
    def compute_gradients_and_update_pre(batch_y0, batch_yN):
        """Takes start positions (x0, y0) and final positions (xN, yN)"""
        pred_y = neural_ode_pre.forward(batch_y0)  # Predict y using Runge-Kutta 4 for each y0 in batch_y0
        with tf.GradientTape() as g_pre:
            g_pre.watch(pred_y)
            loss = tf.reduce_mean(input_tensor=(pred_y - batch_yN)**2) + tf.reduce_sum(input_tensor=tf.abs(model_pre.trainable_weights[0]))
            # This step is computing the loss function
        dLoss = g_pre.gradient(loss, pred_y)  # Here we compute the gradient of the loss function
        h_start, dfdh0, dWeights = neural_ode_pre.backward(pred_y, dLoss)  # Here we compute the dWeights
        optimizer.apply_gradients(zip(dWeights, model_pre.weights))  # Here we update the weights
        return loss, dWeights

    # Compile EAGER graph to static (this will be much faster)
    # compute_gradients_and_update_pre = tfe.defun(compute_gradients_and_update_pre)

    parameters_pre = np.zeros((para_num, 1))

    for step in range(niters_pre):
        print(step)
        loss, dWeights = compute_gradients_and_update_pre(batch_y0, batch_yN)
        parameters_pre = model_pre.trainable_weights[0].numpy()

        print(parameters_pre)

    #########################################
    #          precondition end             #
    #########################################

    # Until here no Bayesian framework is considered. It is just to provide the model with
    # a starting point

    initial_weight = parameters_pre  # We initialize the weights with the parameters found in preconditioning
    print(initial_weight.shape, "here")


    class ODEModel(tf.keras.Model):
        def __init__(self):
            super(ODEModel, self).__init__()
            self.Weights = tf.Variable(tf.random.normal([num_param, 1], dtype=tf.float32)*0.01, dtype=tf.float32)
            # Initializer, initializes the weight to normal random variables with sd = 0.01

        def call(self, inputs, **kwargs):
            t, y = inputs
            h = y
            h1 = h[:, 0:1]
            h2 = h[:, 1:2]

            p1 = self.Weights[0]
            p2 = self.Weights[1]
            p3 = self.Weights[2]  
            p4 = self.Weights[3] 

            h_out1 = p1 * h1 + sigma_y * p2 * h2*h1  # ?? As before, why sigma_y?
            h_out2 = p3 * h2 + sigma_x * p4 * h2*h1  # ?? As before, why sigma_x?
            h_out = tf.concat([h_out1, h_out2], 1)
            return h_out


    model = ODEModel()
    neural_ode = NeuralODE(model, t=t_in)  # We assign to NeuralODE the just created model and the time grid
    # between t0 and t1

    @tf.function
    def compute_gradients_and_update(batch_y0, batch_yN): 
        """Takes start positions (x0, y0) and final positions (xN, yN)"""
        pred_y = neural_ode.forward(batch_y0)  # This finds the predicted yNs
        with tf.GradientTape() as g:
            g.watch(pred_y)
            loss = tf.reduce_sum(input_tensor=(pred_y - batch_yN)**2)  # This creates the loss function
            
        dLoss = g.gradient(loss, pred_y)  # This computes the gradient of the loss function
        h_start, dfdh0, dWeights = neural_ode.backward(pred_y, dLoss)  # This applies the gradient descent to find
        # the updates for the weights

        return loss, dWeights

    # Compile EAGER graph to static (this will be much faster)
    # compute_gradients_and_update = tfe.defun(compute_gradients_and_update)

    #########################################################################################################
    # We now start the Bayesian framework

    # function to compute the kinetic energy, for Hamiltonian Monte Carlo (to be seen with Guglielmi)
    def kinetic_energy(V, loggamma_v, loglambda_v):
        q = (np.sum(-V**2) - loggamma_v**2 - loglambda_v**2)/2.0
        return q

    def compute_gradient_param(dWeights, loggamma, loglambda, batch_size, para_num):
        WW = model.trainable_weights[0].numpy()
        dWeights = np.exp(loggamma)/2.0 * dWeights + np.exp(loglambda) * np.sign(WW)
        return dWeights

    def compute_gradient_hyper(loss, weights, loggamma, loglambda, batch_size, para_num):
        grad_loggamma = np.exp(loggamma) * (loss/2.0 + 1.0) - (batch_size/2.0 + 1.0)
        grad_loglambda = np.exp(loglambda) * (np.sum(np.abs(weights)) + 1.0) - (para_num + 1.0)
        # This somehow computes the gradient of the hyper parameters in order to update them from step to step

        return grad_loggamma, grad_loglambda

    def compute_Hamiltonian(loss, weights, loggamma, loglambda, batch_size, para_num):
        H = np.exp(loggamma)*(loss/2.0 + 1.0) + np.exp(loglambda)*(np.sum(np.abs(weights)) + 1.0)\
                 - (batch_size/2.0 + 1.0) * loggamma - (para_num + 1.0) * loglambda  
        return H

    def leap_frog(v_in, w_in, loggamma_in, loglambda_in, loggamma_v_in, loglambda_v_in):
        # In pratica, non viene utilizzato il gradient descent, ma il leapfrog, algoritmo per ottimizzazione
        # unconstrained implementato in questa porzione di codice... Ci fidiamo...

        model.trainable_weights[0].assign(w_in)
        v_new = v_in
        loggamma_v_new = loggamma_v_in
        loglambda_v_new = loglambda_v_in

        loggamma_new = loggamma_in
        loglambda_new = loglambda_in
        w_new = w_in

        for m in range(L):
            loss, dWeights = compute_gradients_and_update(batch_y0, batch_yN) # evaluate the gradient

            dWeights = np.asarray(dWeights[0]) # make the gradient to be numpy array
            dWeights = compute_gradient_param(dWeights, loggamma_new, loglambda_new, batch_size, para_num)
            grad_loggamma, grad_loglambda = compute_gradient_hyper(loss, w_new, loggamma_new, loglambda_new, batch_size, para_num)

            loggamma_v_new = loggamma_v_new - epsilon/2*grad_loggamma
            loglambda_v_new = loglambda_v_new - epsilon/2*grad_loglambda
            v_new = v_new - epsilon/2*(dWeights)
            w_new = model.trainable_weights[0].numpy() + epsilon * v_new
            model.trainable_weights[0].assign(w_new)
            loggamma_new = loggamma_new + epsilon * loggamma_v_new
            loglambda_new = loglambda_new + epsilon * loglambda_v_new
            
            # Second half of the leap frog
            loss, dWeights = compute_gradients_and_update(batch_y0, batch_yN)
            dWeights = np.asarray(dWeights[0])
            dWeights = compute_gradient_param(dWeights, loggamma_new, loglambda_new, batch_size, para_num)
            grad_loggamma, grad_loglambda = compute_gradient_hyper(loss, w_new, loggamma_new, loglambda_new, batch_size, para_num)

            v_new = v_new - epsilon/2*(dWeights)
            loggamma_v_new = loggamma_v_new - epsilon/2*grad_loggamma
            loglambda_v_new = loglambda_v_new - epsilon/2*grad_loglambda

        print(np.exp(loggamma_new))
        print(np.exp(loglambda_new))

        return v_new, w_new, loggamma_new, loglambda_new, loggamma_v_new, loglambda_v_new

    neural_ode_test = NeuralODE(model, t=t_grid[0:data_size:20])
    parameters = np.zeros((niters, para_num))  # book keeping the parameters
    loggammalist = np.zeros((niters, 1))  # book keeping the loggamma
    loglambdalist = np.zeros((niters, 1))  # book keeping the loggamma
    loglikelihood = np.zeros((niters, 1))  # book keeping the loggamma
    L = 10  # leap frog step number
    epsilon = 0.001  # leap frog step size
    epsilon_max = 0.0002    # max 0.001
    epsilon_min = 0.0002    # max 0.001
    

    def compute_epsilon(step):
        # This will compute the epsilon to use in the leapfrog which is different for every step. It decreases
        # with the steps increasing in number
        coefficient = np.log(epsilon_max/epsilon_min)
        return epsilon_max * np.exp(- step * coefficient / niters)


    # initial weight
    w_temp = initial_weight  # The one we found from the preconditioner
    print("initial_w", w_temp)
    loggamma_temp = 4. + np.random.normal()
    loglambda_temp = np.random.normal()

    model.trainable_weights[0].assign(w_temp)  # We assign to the weights of the model, the ones we found through
    # the pre conditioner. Remember that the initial weights set by the initializer were random distributed according
    # to a Gaussian with 0 mean and 0.01 sd.
    loss_original, _ = compute_gradients_and_update(batch_y0, batch_yN)  # Compute the initial Hamiltonian

    loggamma_temp = np.log(batch_size / loss_original)  # We define an initial guess for loggamma ?? Why defined as such?
    # Michela per Federico: 'precision of the Gaussian noise distribution 'gamma' (vedi pagina 8/22 paper)'

    print("This is initial guess", loggamma_temp, "with loss", loss_original)
    if loggamma_temp > 6.:
        loggamma_temp = 6.
        epsilon_max = 0.0002
        epsilon_min = 0.0002

    # training steps
    for step in range(niters):

        epsilon = compute_epsilon(step)  # Compute the adaptive epsilon for the steps

        print(step)

        v_initial = np.random.randn(para_num, 1)  # initialize the velocity
        loggamma_v_initial = np.random.normal()
        loglambda_v_initial = np.random.normal()

        loss_initial, _ = compute_gradients_and_update(batch_y0, batch_yN) # compute the initial Hamiltonian
        # This line uses the weights of the preconditioner (see line 303) to compute the loss function with those
        loss_initial = compute_Hamiltonian(loss_initial, w_temp, loggamma_temp, loglambda_temp, batch_size, para_num)
        # Then it computes the Hamiltonian

        v_new, w_new, loggamma_new, loglambda_new, loggamma_v_new, loglambda_v_new = \
                                leap_frog(v_initial, w_temp, loggamma_temp, loglambda_temp, loggamma_v_initial, loglambda_v_initial)

        # Then the leapfrog is applied in order to update the parameters and the hyper parameters and to further
        # optimize the weights estimation (L steps of leapfrog at a time)

        # compute the final Hamiltonian
        loss_finial, _ = compute_gradients_and_update(batch_y0, batch_yN)
        loss_finial = compute_Hamiltonian(loss_finial, w_new, loggamma_new, loglambda_new, batch_size, para_num)

        # making decisions
        p_temp = np.exp(-loss_finial + loss_initial + \
                        kinetic_energy(v_new, loggamma_v_new, loglambda_v_new) - kinetic_energy(v_initial, loggamma_v_initial, loglambda_v_initial))

        p = min(1, p_temp)
        p_decision = np.random.uniform()
        if p > p_decision:
            parameters[step:step+1, :] = np.transpose(w_new)  # Parameters are updated
            w_temp = w_new
            loggammalist[step, 0] = loggamma_new
            loglambdalist[step, 0] = loglambda_new
            loglikelihood[step, 0] = loss_finial
            loggamma_temp = loggamma_new
            loglambda_temp = loglambda_new
        else:
            parameters[step:step+1, :] = np.transpose(w_temp)  # New parameters are not updated
            model.trainable_weights[0].assign(w_temp)
            loggammalist[step, 0] = loggamma_temp
            loglambdalist[step, 0] = loglambda_temp
            loglikelihood[step, 0] = loss_initial

        print('probability', p)
        print(p > p_decision)

    print(time.time() - time_in)

    np.save('parameters', parameters)  # The Monte Carlo chain of the parameters
    np.save('loggammalist', loggammalist)  # The Monte Carlo chain of loggamma
    np.save('loglikelihood', loglikelihood)  # The Monte Carlo chain of losses

    np.savetxt("data_weights.csv", parameters, delimiter=',')
    np.savetxt("data_loggammalist.csv", loggammalist, delimiter=',')
    np.savetxt("data_loglikelihood.csv", loglikelihood, delimiter=',')
    np.savetxt("data_loglambda.csv", loglambdalist, delimiter=',')