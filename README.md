# Bayesian data-driven model discovery under uncertainty

Bayesian Statistics Course held @ Politecnico di Milano by Professor Alessandra Guglielmi, Doctor Riccardo Corradin, Doctor Mario Beraha

Developed by: Federico Fatone, Filippo Fedeli, Michela Ceoloni under supervision of Professor Andrea Manzoni.

## Contents 
This project aims to provide a Bayesian framework for model discovery under uncertainty.

In the folder eABCSMC, we define the classes Preconditioner and EABCSMC (for epidemiological models), which constitute the core of the presented e-ABC-SMC method, a new, improved version of the ABC-SMC framework, originally presented in Toni et Al. [2].

An example of their intuitive, end to end usage can be found in the folder Epidemics_response/Examples, where a SIRD/SEIRD model is fitted for Italy and Spain.
All the results contained in the report can be reproduced by running the relevant notebooks/scripts, found in LV for the comparison on simulate data between the framework presented in Perdikaris et al. [1] (LV/Perdikaris) and in Epidemics_Models/Examples for the other relevant results.

The code is developed in Python, for the relevant Python libraries, please refer to requirements.txt and run
''' pip install -r requirements.txt '''
 
If you have any other question, please contact federico.fatone@mail.polimi.it, filippo.fedeli@mail.polimi.it, michela.ceoloni@mail.polimi.it

## Main Results

## Main References

[1] Y. Yang, M.A. Bhouri, P. Perdikaris (2020). Bayesian differential programming for robust systems identification under uncertainty. ArXiv pre-print, submitted to Proceedings of the Royal Society A.

[2] Toni, T., Welch, D., Strelkowa, N., Ipsen, A., Stumpf, M. P. (2009).  Approximate Bayesian Computation scheme for parameter inference and model selection in dynamical systems. Journal of the Royal Society Interface 6(31), 187-202



