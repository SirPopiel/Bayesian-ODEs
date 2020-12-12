setwd('C:\\Users\\PC\\PycharmProjects\\BayNuDes\\BDP_2\\Spiral_noisy')
data <- read.csv('data.csv', sep = ',', header = F)

eq1_coeff <- data[, 1:10]
eq2_coeff <- data[, 11:20]

plot(eq1_coeff)
plot(eq2_coeff)

plot(eq1_coeff[, 1], type = 'l')
plot(eq2_coeff[, 1], type = 'l')

plot(eq1_coeff[, 2], type = 'l')
plot(eq2_coeff[, 2], type = 'l')

plot(eq1_coeff[, 3], type = 'l')
plot(eq2_coeff[, 3], type = 'l')

plot(eq1_coeff[, 4], type = 'l')
plot(eq2_coeff[, 4], type = 'l')

plot(eq1_coeff[, 5], type = 'l')
plot(eq2_coeff[, 5], type = 'l')

# i grafici fanno pena, mettere L = 40 e abbassare/alzare M?
