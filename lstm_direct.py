import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as MSE

import tensorflow as tf
from keras import backend as K

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
    model.add(LSTM(neuronios, input_shape=(lags, 1))) 
    model.add(Dense(1)) # definindo que a saída tem 1 valor na saída
    model.compile(loss='mean_squared_error', optimizer=func_opt)

    return model

def treinar_LSTM(X_treino, y_treino, X_val, y_val, num_exec, idade, o):
    from tensorflow.keras.backend import clear_session
    # Formatação de lags como time-steps
    trainX = X_treino.reshape((X_treino.shape[0], X_treino.shape[1], 1))
    valX = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    iteracoes = 500

    neuronios = [5, 10, 100]
    qtd_lags_sel = X_treino.shape[1]

    rodadas_lista = []
    neuronios_lista = []
    qtd_lag_lista = []
    rmse_lista = []

    batch_size = X_treino.shape[0]
    
    # loop da grid search
    for rodada in range(num_exec):
        melhor_mse = np.Inf # eu zero o melhor_mse para salvar o melhor modelo de cada rodada
        melhor_modelo = None
        for neuronio in neuronios:
            for qtd_lag in range(1, X_treino.shape[1]+1): 
                clear_session()
                lstm = gerar_lstm(neuronio, qtd_lag)
                lstm.fit(trainX[:, -qtd_lag:, :], y_treino, batch_size=batch_size, epochs=iteracoes, verbose=0) # lags como time-steps
                prev_v = lstm.predict(valX[:, -qtd_lag:, :]) # lags como time-steps
                novo_mse = np.sqrt(MSE(y_val, prev_v))

                rodadas_lista.append(rodada)
                neuronios_lista.append(neuronio)
                qtd_lag_lista.append(qtd_lag)
                rmse_lista.append(novo_mse)

                if novo_mse < melhor_mse:   
                    melhor_mse = novo_mse
                    melhor_modelo = lstm
                    qtd_lags_sel = qtd_lag     
                    print('idade:', idade, 'horizonte:',o,'rodada:', rodada,'melhor configuração neuronios:', neuronio, 'qtd_lag:', qtd_lag, 'RMSE:', melhor_mse)
                    lstm.save(f'Resultados - LSTM - Artigo - Horizontes/model_lstm_horizonte_{o}_rodada_{rodada}_idade_{idade}')
                    del lstm 
                    clear_session()
                    gc.collect()
                    
                else:
                    del lstm
                    clear_session()
                    gc.collect()

    # criar df e salvar resultados
    resultados_df = pd.DataFrame({'rodada': rodadas_lista, 'neuronios': neuronios_lista, 'qtd_lag': qtd_lag_lista, 'rmse':rmse_lista})
    resultados_df.to_csv(f'Resultados - LSTM - Artigo - Horizontes/lstm_idade_{idade}_horizonte_{o}_resultados_df.csv')
    # obter o melhor modelo
    melhor_mse_i = resultados_df['rmse'].argmin()
    melhor_rodada = int(resultados_df.iloc[melhor_mse_i]['rodada'])
    melhor_modelo = tf.keras.models.load_model(f'Resultados - LSTM - Artigo - Horizontes/model_lstm_horizonte_{o}_rodada_{melhor_rodada}_idade_{idade}')
    # obter a qtd de lag adequada
    qtd_lags_sel = int(resultados_df.iloc[melhor_mse_i]['qtd_lag'])
    print('QTD. LAGS UTILIZADOS: ', qtd_lags_sel)                  

    return melhor_modelo, qtd_lags_sel       

def prev_lstm(modelo, X_test):
    testX = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_prev = modelo.predict(testX)
    return y_prev


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
    
    # normalizando a série para MLP
    lnmx_norm = normalizar_serie(lnmx.values)
    lnmx_train_norm = lnmx_norm[:-10]
    
    # Separando os dados em X, y e utilizando 2 lags 
    X, y = split_sequence(lnmx_train_norm, n_steps_in=2, n_steps_out = 10)
    X_train, y_train, X_val, y_val  = divisao_dados_temporais(X, y, perc_treino = 0.8)

    # realizar as previsões com 20 modelos, sendo um para cada step
    predictions = np.zeros(10)
    for o in range(10):
        print('-----------------')
        print('HORIZONTE:,', o)
        print('-----------------')

        # treinamento
        lstm_model, lag_sel = treinar_LSTM(X_train, y_train[:,o], X_val, y_val[:,o], 10, idade, o)
        
        # previsao
        X_test_pred = lnmx_train_norm[-lag_sel:].reshape(1, lag_sel)
        lstm_predict = prev_lstm(lstm_model, X_test_pred)
        predictions[o] = lstm_predict

        del lstm_model, lag_sel
        K.clear_session()
        gc.collect()

    # Desnormalizar a série
    lstm_predict = desnormalizar(predictions, lnmx.values)
    previsoes_lstm[:,i] = lstm_predict.values.reshape(10,)
    lstm_predict.index = teste_i.index

    # RMSE
    rmse_lstm[0,i] = np.sqrt(MSE(teste_i, lstm_predict))
    print('-----------------------------------------------')
    print(f'IDADE: {idade}')
    print('RMSE para LSTM = ', np.sqrt(MSE(teste_i, lstm_predict)))
    print('-----------------------------------------------')
    
# Salvando os resultados
previsoes_lstm = pd.DataFrame(previsoes_lstm, columns = idades)
rmse_lstm = pd.DataFrame(rmse_lstm, columns = idades)

# Exportando os resultados para .csv
previsoes_lstm.to_csv('Resultados - LSTM - Artigo - Horizontes/previsoes_lstm_serie.csv', index=None, header=True, encoding = 'utf-8')
rmse_lstm.to_csv('Resultados - LSTM - Artigo - Horizontes/rmse_lstm_serie.csv', index=None, header=True, encoding = 'utf-8')

