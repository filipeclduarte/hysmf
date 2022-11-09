# Importação das bibliotecas
# Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error as MSE
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.pylab import rcParams

import math
import itertools
import optunity
import optunity.metrics

# Criando os conjuntos de treinamento, validação e teste
def divisao_dados_temporais(X,y, perc_treino, perc_val = 0):
    tam_treino = int(perc_treino * len(y))
    
    if perc_val > 0:        
        tam_val = int(len(y)*perc_val)
              
        X_treino = X[0:tam_treino,:]
        y_treino = y[0:tam_treino,:]
        
        print("Particao de Treinamento:", 0, tam_treino)
        
        X_val = X[tam_treino:tam_treino+tam_val,:]
        y_val = y[tam_treino:tam_treino+tam_val,:]
        
        print("Particao de Validacao:",tam_treino,tam_treino+tam_val)
        
        X_teste = X[(tam_treino+tam_val):-1,:]
        y_teste = y[(tam_treino+tam_val):-1,:]
        
        print("Particao de Teste:", tam_treino+tam_val, len(y))
        
        return X_treino, y_treino, X_teste, y_teste, X_val, y_val
        
    else:
        
        X_treino = X[0:tam_treino,:]
        y_treino = y[0:tam_treino,:]

        X_teste = X[tam_treino:-1,:]
        y_teste = y[tam_treino:-1,:]

        return X_treino, y_treino, X_teste, y_teste 

def calculate_std_arima(arima_mean, arima_confidence_interval):
    '''
    Calcula com confiança de 95%
    '''
    # IC = media +- 1.96 * desvio
    # Desvio = (LS - media)/1.96

    desvio_arima = (arima_confidence_interval[:,1] - arima_mean)/1.96

    return desvio_arima

# importando os dados
lnmx_series = pd.read_csv('lnmx_series.csv', sep = ',', index_col = 0)

# Separar os dados em treinamento e teste
# treinamento: 1816-1990
# teste: 1991-2010
# treino = lnmx_series.loc[1949:1990,:]
treino = lnmx_series.loc[1950:2008, :]
# teste = lnmx_series.loc[1991:2010, :]
teste = lnmx_series.loc[2009:2018, :]
# lnmx_series = lnmx_series.loc[1950:2011,:]
lnmx_series = lnmx_series.loc[1950:2018, :]

# idades utilizadas
idades = treino.columns

previsoes_arima = np.zeros((10,20)) 
previsoes_arima_std = np.zeros((10, 20))
rmse_arima = np.zeros((1,20))


# previsoes_arima = np.zeros((20,20)) 
# previsoes_arima_std = np.zeros((20, 20))
# rmse_arima = np.zeros((1,20))

# nome_grafico = ['idade_' + str(idade) for idade in idades]
# label_grafico = ['Série da Idade ' + str(idade) for idade in idades]

# iteração para realizar o procedimento para cada idade
for idade, i in zip(idades, range(0,len(idades)+1)):
    lnmx = lnmx_series[idade]
    treino_i = treino[idade]
    teste_i = teste[idade]
    
    # ARIMA
    arima = auto_arima(treino_i, start_p=1, start_q=1,
                           max_p=10, max_q=10, m=12,
                           start_P=0, seasonal=False,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
    
    # treinamento
    arima.fit(treino_i)
    
    # erro = resíduos do modelo arima
    arima_error = arima.resid()

    # previsao arima - 21 períodos porque quando fizer a separação dos dados, vamos perder a última obs.
    # arima_predict, confidence_interval_predict = arima.predict(n_periods=21, return_conf_int=True)
    arima_predict, confidence_interval_predict = arima.predict(n_periods=10, return_conf_int=True)

    # calcular desvio padrão a partir do intervalo de confiança
    arima_std = calculate_std_arima(arima_predict, confidence_interval_predict)

    # transformando em série
    previsoes_arima[:,i] = arima_predict
    # arima_predict = pd.Series(arima_predict, index = lnmx.loc[1991:2011].index)
    arima_predict = pd.Series(arima_predict, index = lnmx.loc[2009:2018].index)

    previsoes_arima_std[:, i] = arima_std


    # Calculando o erro de previsao do arima 
    # arima_error_predict = lnmx.loc[1991:] - arima_predict
    arima_error_predict = lnmx.loc[2009:] - arima_predict

# RMSE
    rmse_arima[0,i] = np.sqrt(MSE(teste_i, arima_predict.loc[:2018]))
    print('-----------------------------------------------')
    print('Ln(mx) para a idade ', idade, 'RMSE - ARIMA = ',np.sqrt(MSE(teste_i, arima_predict.loc[:2018])))
    print('-----------------------------------------------')

# Salvando os resultados
previsoes_arima = pd.DataFrame(previsoes_arima, columns= idades)
previsoes_arima_std = pd.DataFrame(previsoes_arima_std, columns = idades)
rmse_arima = pd.DataFrame(rmse_arima, columns = idades)

# Exportando os resultados para .csv
previsoes_arima.to_csv('previsoes_arima.csv', index=None, header=True, encoding = 'utf-8')
previsoes_arima_std.to_csv('previsoes_arima_std.csv', index = None, header = True, encoding = 'utf-8')
rmse_arima.to_csv('rmse_arima.csv', index= None, header= True, encoding= 'utf-8')