reg_code <- 5   # 3 = Lombardia, 1 = Piemonte, 5 = Veneto, 14 = Molise, 8 = Emilia-Romagna, 12 = Lazio

setwd("~/GitHub/BayNuDes/BDP_2/SIR")

covid <- read.csv("C:/Users/fedfa/OneDrive/Desktop/covid_data_regions.csv")
I <- covid$totale_positivi[which(covid$codice_regione==reg_code)]
plot(1:length(I), I)
R <- covid$dimessi_guariti[which(covid$codice_regione==reg_code)]
plot(1:length(R), R)
D <- covid$deceduti[which(covid$codice_regione==reg_code)]
plot(1:length(D), D)

region <- data.frame(I = I, R = R, D = D)
write.csv(region, 'RegionER.csv', row.names = FALSE)

x.grid <- seq(0, 04, length.out=1000)
plot(x.grid, dgamma(x.grid, 2*0.5, 0.5), type='l')
