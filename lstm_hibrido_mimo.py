# HÍBRIDO ARIMA COM LSTM

# Importação das bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_squared_error as MSE

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

# 1 modelo - 20 saídas
def gerar_lstm(neuronios,  lags, func_opt='adam'):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, GRU

    model = Sequential()
    model.add(LSTM(neuronios, input_shape=(lags,1))) 
    # model.add(GRU(neuronios, input_shape=(lags,1)))
    model.add(Dense(10)) # definindo que a saída tem 10 valores
    model.compile(loss='mean_squared_error', optimizer=func_opt)

    return model        

def treinar_LSTM(X_treino, y_treino, X_val, y_val, num_exec, execucao, idade):
    from tensorflow.keras.backend import clear_session

    # estruturar dados
    trainX = np.reshape(X_treino, (X_treino.shape[0], X_treino.shape[1], 1))
    valX = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

    iteracao = 100
    # iteracao = 100
    # neuronios = [5, 10, 100, 500]
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
                # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                # lstm.fit(trainX[:,-qtd_lag:, :], y_treino, epochs=iteracao, callbacks=[callback], 
                #         validation_data=(valX[:,-qtd_lag:, :], y_val), 
                #         batch_size = batch_size, verbose=0) # testar com mais epochs (100, 500, 1000)
                lstm.fit(trainX[:,-qtd_lag:, :], y_treino, epochs=iteracao,
                        batch_size=batch_size, verbose=0)
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
                    print('Execução:',execucao,'idade:', idade,'rodada:', rodada,'melhor configuração neuronios:', neuronio, 'qtd_lag:', qtd_lag, 'RMSE:', melhor_mse)
                    lstm.save(f'Resultados - Híbrido - ARIMA - LSTM_exec_{execucao}/model_lstm_rodada_{rodada}_idade_{idade}')
                    del lstm 
                    clear_session()
                    gc.collect()
                else:
                    del lstm
                    clear_session()
                    gc.collect()

    # criar df e salvar resultados
    resultados_df = pd.DataFrame({'rodada': rodadas_lista, 'neuronios': neuronios_lista, 'qtd_lag': qtd_lag_lista, 'iteracoes': iteracoes_lista,'rmse':rmse_lista})
    resultados_df.to_csv(f'Resultados - Híbrido - ARIMA - LSTM_exec_{execucao}/lstm_idade_{idade}_resultados_df.csv')
    # ler o melhor modelo da grid
    melhor_mse_i = resultados_df['rmse'].argmin()
    melhor_rodada = int(resultados_df.iloc[melhor_mse_i]['rodada'])
    melhor_modelo = tf.keras.models.load_model(f'Resultados - Híbrido - ARIMA - LSTM_exec_{execucao}/model_lstm_rodada_{melhor_rodada}_idade_{idade}')
    # obter a qtd de lag adequada
    qtd_lags_sel = int(resultados_df.iloc[melhor_mse_i]['qtd_lag'])
    print('QTD. LAGS UTILIZADOS: ', qtd_lags_sel)

    return melhor_modelo, qtd_lags_sel 

def prev_lstm(modelo, X_test):
    testX = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_prev = modelo.predict(testX)
    return y_prev

def main(qtd_execucoes=1):
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

    for execucao in range(1, qtd_execucoes+1):
    # for execucao in range(1, 11):
        print(f'Execução: {execucao}')
        np.random.seed(execucao + 5)
        tf.random.set_seed(execucao + 5)

        previsoes_arima = np.zeros((10,20)) 
        previsoes_error_lstm = np.zeros((10,20))
        previsoes_hibrido = np.zeros((10,20))
        rmse_arima = np.zeros((1,20))
        rmse_error_lstm = np.zeros((1,20))
        rmse_hibrido = np.zeros((1,20))

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
            arima_predict = arima.predict(n_periods=10, return_conf_int=False)
            previsoes_arima[:,i] = arima_predict
            arima_predict = pd.Series(arima_predict, index = lnmx.loc[2009:2018].index)
            
            # Calculando o erro de previsao do arima 
            arima_error_predict = lnmx.loc[2009:] - arima_predict

            # juntando os erros do arima treinamento e teste
            arima_error_train_test = np.concatenate((arima_error, arima_error_predict))

            # normalizar erro
            arima_error_train_test_norm = normalizar_serie(arima_error_train_test)
            arima_error_norm = arima_error_train_test_norm[:-10]

            # Separando os dados em X, y e utilizando 2 lags 
            X_error, y_error = split_sequence(arima_error_norm, n_steps_in=2, n_steps_out = 10)
            X_train_error, y_train_error, X_val_error, y_val_error  = divisao_dados_temporais(X_error, y_error, perc_treino = 0.8)

            # treinar LSTM no erro do arima
            lstm_model_error, lag_sel_error = treinar_LSTM(X_train_error, y_train_error, X_val_error, y_val_error, 10, execucao, idade)
              
            # Realizar a previsão
            ### Ajustar a formatação da série
            # X_test_pred = arima_error_train_test_norm[-(lag_sel_error + 20):-20].reshape(1, lag_sel_error)
            X_test_pred = arima_error_norm[-lag_sel_error:].reshape(1, lag_sel_error)
            lstm_predict_error = prev_lstm(lstm_model_error, X_test_pred).reshape(10, )

            del lstm_model_error
            K.clear_session()
            gc.collect()
            
            # transformar em série
            # lstm_predict_error = pd.Series(lstm_predict_error.reshape(20,), index = teste.index)
            
            # Desnormalizar a série
            lstm_predict_error = desnormalizar(lstm_predict_error, arima_error_train_test)
            
            # previsoes_error_lstm[:,i] = lstm_predict_error.values.reshape(1,-1)
            previsoes_error_lstm[:, i] = lstm_predict_error.values.reshape(10,)
            # lstm_predict_error = lstm_predict_error[0]
            lstm_predict_error.index = teste_i.index 

            #Criando o sistema híbrido 
            # z = arima_predict.loc[:2010] + lstm_predict_error
            z = arima_predict.loc[:2018]  + lstm_predict_error[0]
            # z = pd.Series(z, index = teste.index)            
            # previsoes_hibrido[:,i] = z 
            previsoes_hibrido[:,i] = z.values
            
            # RMSE
            rmse_arima[0,i] = np.sqrt(MSE(teste_i, arima_predict.loc[:2018]))
            rmse_error_lstm[0,i] = np.sqrt(MSE(arima_error_predict, lstm_predict_error))
            rmse_hibrido[0,i] = np.sqrt(MSE(teste_i, z))
            print('-----------------------------------------------')
            print('Ln(mx) para a idade ', idade, 'RMSE - ARIMA = ',np.sqrt(MSE(teste_i, arima_predict.loc[:2018])))
            print('RMSE para o Modelo híbrido Z = ARIMA + LSTM = ', np.sqrt(MSE(teste_i, z)))
            print('-----------------------------------------------')

        # Salvando os resultados
        previsoes_error_lstm = pd.DataFrame(previsoes_error_lstm, columns = idades)
        previsoes_hibrido = pd.DataFrame(previsoes_hibrido, columns = idades)
        rmse_error_lstm = pd.DataFrame(rmse_error_lstm, columns = idades)
        rmse_hibrido = pd.DataFrame(rmse_hibrido, columns = idades)

        # Exportando os resultados para .csv
        previsoes_error_lstm.to_csv(f'Resultados - Híbrido - ARIMA - LSTM_exec_{execucao}/previsoes_error_lstm_exec_{execucao}.csv', index=None, header=True, encoding = 'utf-8')
        previsoes_hibrido.to_csv(f'Resultados - Híbrido - ARIMA - LSTM_exec_{execucao}/previsoes_hibrido_exec_{execucao}.csv', index=None, header=True, encoding = 'utf-8')
        rmse_error_lstm.to_csv(f'Resultados - Híbrido - ARIMA - LSTM_exec_{execucao}/rmse_error_lstm_exec_{execucao}.csv', index=None, header=True, encoding = 'utf-8')
        rmse_hibrido.to_csv(f'Resultados - Híbrido - ARIMA - LSTM_exec_{execucao}/rmse_hibrido_exec_{execucao}.csv', index=None, header=True, encoding = 'utf-8')
            

if __name__ == '__main__':
    print('Iniciando o script Híbrido ARIMA-LSTM')
    main()
    print('FIM!')