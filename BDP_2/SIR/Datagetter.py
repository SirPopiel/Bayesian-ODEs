import pandas as pd 

def get_data(country_code = 'IT', start_date = '2020-02-24', end_date = '2020-12-25', source = 'JHU', model = 'SIRD', rdata = True):
    
    if rdata:
        import pyreadr 
        result = pyreadr.read_r('Data\COVID-19.RData')
        covid = result["COVID19"] 
        contagio = covid[covid.ID == country_code]
        contagio.Date = pd.to_datetime(contagio.Date, infer_datetime_format=True)  
    else:
        contagio = pd.read_csv('Data/COVID_' + country_code + '.csv')
    contagio.drop(columns = ['ID','Age','Sex'], inplace = True)
    contagio = contagio[(contagio['Date'] >= start_date) & (contagio['Date'] <= end_date)]
    contagio = contagio[contagio['Source']=='JHU']

    dates = []
    infected = []
    dead = []
    recovered = []
    
    try :
        for i in contagio.Date.unique():
            dates.append(contagio[(contagio['Date'] == i) & (contagio['Type'] == 'Active')].Date.item())
            infected.append(contagio[(contagio['Date'] == i) & (contagio['Type'] == 'Active')].Cases.item())
            dead.append(contagio[(contagio['Date'] == i) & (contagio['Type'] == 'Deaths')].Cases.item())
            recovered.append(contagio[(contagio['Date'] == i) & (contagio['Type'] == 'Recovered')].Cases.item())
        
    except:
        print("Error: check wether the required source and dates are present for this country.")
        return -1
    
    sird = pd.DataFrame(list(zip(dates, infected, dead, recovered)), columns = ['Date','Infected','Dead','Recovered'])
    return sird

    