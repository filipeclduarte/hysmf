import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error as MSE

import math
import itertools
import optunity
import optunity.metrics

import tensorflow as tf
from keras import backend as K
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import gc


# Funções essenciais para a organização dos dados
def normalizar_serie(serie):
    minimo = min(serie)
    maximo = max(serie)
    y = (serie - minimo) / (maximo - minimo)
    return y

def desnormalizar(serie_atual, serie_real):
    minimo = min(serie_real)
    maximo = max(serie_real)
    
    serie = (serie_atual * (maximo - minimo)) + minimo
    
    return pd.DataFrame(serie)

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

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


def gerar_lstm(neuronios,  lags, func_opt='adam'):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense

    model = Sequential()
    model.add(LSTM(neuronios, input_shape=(lags,1))) 
    model.add(Dense(10)) # definindo que a saída tem 10 valores
    model.compile(loss='mean_squared_error', optimizer=func_opt)

    return model        


def treinar_LSTM(X_treino, y_treino, X_val, y_val, num_exec, execucao, idade):
    from tensorflow.keras.backend import clear_session

    # estruturar dados
    trainX = np.reshape(X_treino, (X_treino.shape[0], X_treino.shape[1], 1))
    valX = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

    iteracao = 500
    neuronios = [5, 10, 100]
    # func_opt = 'adam'
    # melhor_modelo = None
    qtd_lags_sel = X_treino.shape[1]
    batch_size = X_treino.shape[0]

    # criar listas para armazenar os resultados da grid
    rodadas_lista = []
    neuronios_lista = []
    qtd_lag_lista = []
    iteracoes_lista = []
    rmse_lista = []
    
    # loop da grid search
    for rodada in range(num_exec):      
        melhor_mse = np.Inf # eu zero o melhor_mse para cada rodada 
        melhor_modelo = None      
        for neuronio in neuronios:
            for qtd_lag in range(1, X_treino.shape[1]+1):
                clear_session()
                lstm = gerar_lstm(neuronio, qtd_lag)
                lstm.fit(trainX[:,-qtd_lag:, :], y_treino, epochs=iteracao, batch_size = batch_size, verbose=0)
                prev_v = lstm.predict(valX[:, -qtd_lag:, :])

                novo_mse = np.sqrt(MSE(y_val, prev_v))
                
                # atualizacao das listas
                rodadas_lista.append(rodada)
                neuronios_lista.append(neuronio)
                qtd_lag_lista.append(qtd_lag)
                iteracoes_lista.append(iteracao)
                rmse_lista.append(novo_mse)
                
                # Avaliação do melhor modelo, armazenamento e liberação da memória
                if novo_mse < melhor_mse:   
                    melhor_mse = novo_mse
                    melhor_modelo = lstm
                    qtd_lags_sel = qtd_lag                                      
                    print('Execução:',execucao,'idade:', idade,'rodada:', rodada,
                          'melhor configuração neuronios:', neuronio, 
                          'qtd_lag:', qtd_lag, 'RMSE:', melhor_mse)
                    lstm.save(f'Resultados - LSTM - Artigo - MIMO/exec_{execucao}_model_lstm_rodada_{rodada}_idade_{idade}')
                    del lstm 
                    clear_session()
                    gc.collect()
                else:
                    del lstm
                    clear_session()
                    gc.collect()

    # criar df e salvar resultados
    resultados_df = pd.DataFrame({'rodada': rodadas_lista, 'neuronios': neuronios_lista, 'qtd_lag': qtd_lag_lista, 
                                  'iteracoes': iteracoes_lista,'rmse':rmse_lista})
    resultados_df.to_csv(f'Resultados - LSTM - Artigo - MIMO/lstm_idade_{idade}_resultados_df.csv')
    # ler o melhor modelo da grid
    melhor_mse_i = resultados_df['rmse'].argmin()
    melhor_rodada = int(resultados_df.iloc[melhor_mse_i]['rodada'])
    melhor_modelo = tf.keras.models.load_model(f'Resultados - LSTM - Artigo - MIMO/exec_{execucao}_model_lstm_rodada_{melhor_rodada}_idade_{idade}')
    # obter a qtd de lag adequada
    qtd_lags_sel = int(resultados_df.iloc[melhor_mse_i]['qtd_lag'])
    print('QTD. LAGS UTILIZADOS: ', qtd_lags_sel)

    return melhor_modelo, qtd_lags_sel 


def prev_lstm(modelo, X_test):
    testX = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_prev = modelo.predict(testX)
    return y_prev

np.random.rand(6)
tf.random.set_seed(6)
# importando os dados
lnmx_series = pd.read_csv('lnmx_series.csv', sep = ',', index_col = 0)

# Separar os dados em treinamento e teste
# treinamento: 1816-1990
# teste: 1991-2010
treino = lnmx_series.loc[1950:2008,:]
teste = lnmx_series.loc[2009:2018, :]
lnmx_series = lnmx_series.loc[1950:2018,:]

# idades utilizadas
idades = treino.columns

previsoes_lstm = np.zeros((10,20))
rmse_lstm = np.zeros((1,20))

# iteração para realizar o procedimento para cada idade
for idade, i in zip(idades, range(0,len(idades)+1)):
    lnmx = lnmx_series[idade]
    treino_i = treino[idade]
    teste_i = teste[idade]
    print(f'IDADE: {idade}')
    
    # normalizando a série para MLP
    lnmx_norm = normalizar_serie(lnmx.values)
    treino_i_norm = lnmx_norm[:-10]

    # Separando os dados em X, y e utilizando 2 lags 
    X, y = split_sequence(treino_i_norm, n_steps_in=2, n_steps_out = 10)
    X_train, y_train, X_val, y_val  = divisao_dados_temporais(X, y, perc_treino = 0.8)

    # treinar LSTM
    lstm_model, lag_sel = treinar_LSTM(X_train, y_train, X_val, y_val, 10, 1, idade)
    
    # Realizar a previsão
    # lstm_predict = prev_lstm(lstm_model, X_test[-1, -len(X_test[0]):].reshape(1, len(X_test[0])))
    X_test_pred = treino_i_norm[-lag_sel:].reshape(1, lag_sel)
    lstm_predict = prev_lstm(lstm_model, X_test_pred).reshape(10,)

    del lstm_model
    K.clear_session()
    gc.collect()

    # desnormalizando a previsão
    lstm_predict = desnormalizar(lstm_predict, lnmx.values)

    # armazenando previsoes
    previsoes_lstm[:, i] = lstm_predict.values.reshape(10,)
    lstm_predict.index = teste_i.index

    # RMSE
    rmse_lstm[0,i] = np.sqrt(MSE(teste_i, lstm_predict))
    print('-----------------------------------------------')
    print('RMSE para LSTM = ', np.sqrt(MSE(teste_i, lstm_predict)))
    print('-----------------------------------------------')
    

# Salvando os resultados
previsoes_lstm = pd.DataFrame(previsoes_lstm, columns = idades)
rmse_lstm = pd.DataFrame(rmse_lstm, columns = idades)

# Exportando os resultados para .csv
previsoes_lstm.to_csv('Resultados - LSTM - Artigo - MIMO/previsoes_lstm_serie.csv', index=None, header=True, encoding = 'utf-8')
rmse_lstm.to_csv('Resultados - LSTM - Artigo - MIMO/rmse_lstm_serie.csv', index=None, header=True, encoding = 'utf-8')

