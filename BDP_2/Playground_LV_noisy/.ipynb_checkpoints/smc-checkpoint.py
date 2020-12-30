

def adaptive_gaussian_sampling(true_center, loc, scale, quantile_1, quantile_2):
    if ((sps.norm.cdf(true_center, loc=loc, scale=scale) < quantile_1) | (sps.norm.cdf(true_center, loc=loc, scale=scale) > quantile_2)):
        par = npr.uniform(0,1)
        a = 0
        b = 0
        if true_center > loc:
            a = ((loc + par*(true_center-loc)) - loc)/scale
            b = float('inf')
            w_temp = sps.truncnorm.rvs(a, b, loc=loc, scale=scale)
        else:
            a = ((loc + par*(true_center-loc)) - loc)/scale
            b = -float('inf')
            w_temp = sps.truncnorm.rvs(b, a, loc=loc, scale=scale)
    else:
        w_temp = npr.normal(loc, scale)
    
    return(w_temp)

def border_estimates_e_abc(model, temp_model, neural_ode_temp, eps = 0.075, niters = 5000,  n_cores = 1):
    import ray
    import scipy.stats as sps
    
    # This function estimates the borders of the region containing the eps-approximate posterior through niters 
    #  iterations. It uses the empirical sampling method to get nearer to the correct acceptance region.
    #  This is only used to estimate borders, though it can estimate also the parameters in order to avoid 
    #  biases in the estimation.

    #model.trainable_weights[0].assign(parameters_pre)
    initial_loss, _ = compute_gradients_and_update(batch_y0, batch_yN)   # The result we obtained from the simulation of the model
    initial_loss = 0
    
    @ray.remote(num_returns = 2)
    def sampling(niters,n_params, verbose = 0):
    
        
        loss = []    
        parameters = np.zeros((niters, para_num))  # book keeping the parameters
        lambdalist = np.zeros((niters, 1))  # book keeping the loggamma
        naccepted = 0
        
        for i in tqdm(range(niters)) if verbose else range(niters):
            
            w_temp = np.zeros(n_params)
            lambda_temp = npr.uniform(low = 0, high = 1.5) # come scelgo high?
            WW = model.trainable_weights[0].numpy()  # passare model?
            alpha_quant = 0.25
            
            # si può anche adottare uno schema di parallelizzazione con i parametri in realtà
            
            w_temp = np.resize([adaptive_gaussian_sampling(w_temp[i],0,1/lambda_temp,alpha_quant,1-alpha_quant) for i in range(n_params)],(n_params,1)) 
            # perché passo sia quantile sup che inf??
            temp_model.trainable_weights[0].assign(w_temp) # passare temp_model?
            
            sim_loss = compute_loss(batch_y0, batch_yN, neural_ode_temp) # ma qua va guardato modello nuovo senza aggiornare
            loss.append(sim_loss)

            if np.abs(initial_loss - sim_loss) < (eps + npr.uniform()*eps*1/3): # passare initial loss
                parameters[naccepted:naccepted+1, :] = np.transpose(w_temp)
                lambdalist[naccepted:naccepted+1] = lambda_temp
                naccepted += 1
            
            return parameters, naccepted #,lambdalist, -> acchecceserve
    

    
    ray.init(ignore_reinit_error=True)
    parameters, naccepted = ray.get([sampling.remote(n_iters//n_cores + (1 if n_iters%i<n_iters%n_cores else 0) ,n_params, verbose = (1 if i==0 else 0) ) for i in range(n_cores)])
    naccepted = np.sum(naccepted)
    print(parameters.shape)

    print('Acceptance rate: ', naccepted / niters)

    parameters = parameters[0:naccepted,:]
    lambdalist = lambdalist[0:naccepted,:]
    
    # attenzione, problem specific
    border1 = np.min(parameters[:,0])
    border2 = np.max(parameters[:,1])
    border3 = np.max(parameters[:,2])
    border4 = np.min(parameters[:,3])

    return([border1, border2, border3, border4])