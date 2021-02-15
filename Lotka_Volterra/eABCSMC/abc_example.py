import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import numpy.random as npr
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('agg')
#import tensorflow.contrib.eager as tfe
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
                       #  data found through the integration of the system)
    batch_time = 320  #
    niters = 1000  # Number of Hamiltonian MC iterations
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
    #  for the trajectories

    true_y = true_yy

    # normalizing data and system
    sigma_x = np.std(true_yy[:, 0:1])  # Standard deviation of data for x = preys
    sigma_y = np.std(true_yy[:, 1:2])  # Standard deviation of data for y = predators

    noise_level = 0.03  # Adding some nois with this noise level
    sigma_normal = max(sigma_x, sigma_y)  # Considering the highest between the 2

    true_y[:, 0:1] = true_y[:, 0:1]/sigma_x + noise_level * np.random.randn(true_y[:, 0:1].shape[0], true_y[:, 0:1].shape[1])
    true_y[:, 1:2] = true_y[:, 1:2]/sigma_y + noise_level * np.random.randn(true_y[:, 1:2].shape[0], true_y[:, 1:2].shape[1])
    # The 2 lines above normalize the data and then add a noise being extract from a gaussian
    #  random variable with 0 mean and (noise_level)^2 variance.


    def get_batch():
        """Returns initial point and last point over sampled frament of trajectory"""
        starts = np.random.choice(np.arange(data_size - batch_time - 1, dtype=np.int64), batch_size, replace=False)
        # This randomly chooses from {0, 1, ... , data_size - batch_time - 1}, batch_size different elements
        batch_y0 = true_y[starts] 
        batch_yN = true_y[starts + batch_time]
        # The function returns a tensor composed by some y0 and the respective yN,
        #  being y0 + DeltaT.
        return tf.cast(batch_y0, dtype=tf.float32), tf.cast(batch_yN, dtype=tf.float32)


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
            #  the actual weights of the model
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
    #  a starting point 

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


##########################################################################################################################

# DA QUI IN POI NUOVO

# Qui sotto parte una bozza di implementazione del metodo ABC come descritto su
# https://en.wikipedia.org/wiki/Approximate_Bayesian_computation

# In particolare quello che si fa è:
# 1. Sampling dalla prior
# 2. Generazione dell'output D_hat dal modello ottenuto con i parametri theta_hat campionati in 1.
# 3. Calcolo di distance(D_hat, D), dove D è l'output ottenuto dai dati veri
# 4. Se distance(D_hat, D) < epsilon tengo theta_hat, altrimenti butto via e rifaccio

##########################################################################################################################

    # We now start the Bayesian framework
    # We will compute the Approximate Bayesian Computation approximation from the posterior

    initial_loss = ...   # The result we obtained from the simulation of the model
    # In teoria questa loss è un minimo locale, quindi dovrebbe essere la più piccola. Più si è vicini a questa
    # loss con i theta individuati e meglio è per la simulazione.
    # In alternativa alla loss, per la quale non riesco a trovare giustificazioni teoriche assolutamente convincenti
    # si possono prendere le traiettorie e calcolare lo scostamento, questo sarebbe un approccio "esatto", ma computazionalmente
    # sbatti...
    # initial_trajectories = ...
    naccepted = 0
    accepted = False
    eps = ...  # To be determined, it is a hyperparameter
    parameters = np.zeros((niters, para_num))  # book keeping the parameters
    lambdalist = np.zeros((niters, 1))  # book keeping the loggamma

    lambda_sim = 0
    w_sim = 0

    for i in range(niters):
        print("Iteration: ", i)
        # We will consider here the standard Lasso model for data approximation
        lambda_temp = npr.gamma(para_num+1, 1)
        WW = model.trainable_weights[0].numpy()
        w_temp = WW + npr.laplace(0, scale=1/lambda_sim, size=WW.size)

        # After the sampling from the prior we go on by simulating the model
        # sim_trajectories = simulate_trajectories_from_theta_hat(model, w_temp)
        sim_loss = compute_loss(model, w_temp)

        # if abc_distance(initial_trajectories, sim_trajectories) < eps:
        if np.abs(initial_loss - sim_loss) < eps:
            parameters[i:i+1, :] = np.transpose(w_temp)
            lambdalist[i:i+1] = lambda_temp
            lambda_sim = lambda_temp
            w_sim = w_temp
            naccepted += 1
            accepted = True

        else:
            parameters[i:i + 1, :] = np.transpose(w_sim)
            lambdalist[i:i + 1] = lambda_sim
            accepted = False

        print("Accepted = ", accepted)

    print('Acceptance rate: ', naccepted / niters)

    np.save('parameters', parameters)  # The Monte Carlo chain of the parameters
    np.save('lambda', lambdalist)

    np.savetxt("data_weights.csv", parameters, delimiter=',')
    np.savetxt("data_lambda.csv", lambdalist, delimiter=',')
