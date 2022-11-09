from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
from pmdarima.arima import auto_arima
import tensorflow as tf
from tensorflow.keras import backend as K
from utils import normalizar_serie, desnormalizar, split_sequence, divisao_dados_temporais

from nbeats_keras.model import NBeatsNet as NBeatsKeras

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import gc



def build_model(
    time_steps,
    output_dim,
    hidden_units=64
):
    backend = NBeatsKeras(
            backcast_length=time_steps, forecast_length=output_dim,
            stack_types=(NBeatsKeras.GENERIC_BLOCK, NBeatsKeras.GENERIC_BLOCK),
            nb_blocks_per_stack=2, thetas_dim=(4, 4), share_weights_in_stack=True,
            hidden_layer_units=hidden_units
        )

        # Definition of the objective function and the optimizer.
    backend.compile(loss='mean_squared_error', optimizer='adam',metrics=["mean_squared_error"])
    return backend


def treinar_nbeats(X_treino, y_treino, X_val, y_val, num_exec, execucao, idade, o):
    from tensorflow.keras.backend import clear_session
    
    # Formatação de lags como time-steps
    trainX = X_treino.reshape((X_treino.shape[0], X_treino.shape[1], 1))
    valX = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    # iteracoes = [500]
    iteracoes = [100]
    neuronios = [64, 128]
    # neuronios = [5, 10, 50, 100]
    qtd_lags_sel = X_treino.shape[1]

    rodadas_lista = []
    neuronios_lista = []
    iteracoes_lista = []
    rmse_lista = []

    batch_size = X_treino.shape[0]
    
    # loop da grid search
    for rodada in range(num_exec):
        melhor_mse = np.Inf # eu zero o melhor_mse para salvar o melhor modelo de cada rodada
        melhor_modelo = None
        print(f'Rodada: {rodada}')
        for neuronio in neuronios:
            print(f'Neurônios: {neuronio}')
            for iteracao in iteracoes:
                clear_session()
                model = build_model(qtd_lags_sel, 1, neuronio)
                callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

                model.fit(
                    trainX,
                    y_treino,
                    validation_data = (valX, y_val),
                    epochs=iteracao,
                    batch_size=batch_size,
                    callbacks=callbacks,
                        )
                
                # lags como time-steps
                prev_v = np.squeeze(model.predict(valX),axis=2) # remover a 3 dim
                novo_mse = np.sqrt(MSE(y_val, prev_v))

                rodadas_lista.append(rodada)
                neuronios_lista.append(neuronio)
                iteracoes_lista.append(iteracao)
                rmse_lista.append(novo_mse)

                if novo_mse < melhor_mse:   
                    melhor_mse = novo_mse
                    melhor_modelo = model
                    print('Execução:',execucao,'idade:', idade,'horizonte', o, 'rodada:', rodada,'melhor configuração neuronios:', neuronio, 'RMSE:', melhor_mse)
                    model.save(f'Resultados - hibrido - n-beats - Artigo - direct/model_n_beats_horizonte_{o}_rodada_{rodada}_idade_{idade}')
                    del model
                    clear_session()
                    gc.collect()
                    
                else:
                    del model
                    clear_session()
                    gc.collect()
                    

    # criar df e salvar resultados
    resultados_df = pd.DataFrame({'rodada': rodadas_lista, 'neuronios': neuronios_lista, 'iteracoes': iteracoes_lista,'rmse':rmse_lista})
    resultados_df.to_csv(f'Resultados - hibrido - n-beats - Artigo - direct/n_beats_idade_{idade}_horizonte_{o}_resultados_df.csv')
    # obter o melhor modelo
    melhor_mse_i = resultados_df['rmse'].argmin()
    melhor_rodada = int(resultados_df.iloc[melhor_mse_i]['rodada'])
    melhor_modelo = tf.keras.models.load_model(f'Resultados - hibrido - n-beats - Artigo - direct/model_n_beats_horizonte_{o}_rodada_{melhor_rodada}_idade_{idade}')
    
    return melhor_modelo

def prev_n_beats(modelo, X_test):
    testX = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_prev = modelo.predict(testX)
    return y_prev



def main(qtd_execucoes=1):
    # importando os dados
    lnmx_series = pd.read_csv('lnmx_series.csv', sep = ',', index_col = 0)

    # Separar os dados em treinamento e teste
    treino = lnmx_series.loc[1950:2008,:]
    teste = lnmx_series.loc[2009:2018, :]
    lnmx_series = lnmx_series.loc[1950:2018,:]

    # idades utilizadas
    idades = treino.columns

    for execucao in range(1, qtd_execucoes+1):
        print(f'Execução: {execucao}')
        np.random.seed(execucao + 5)
        tf.random.set_seed(execucao + 5)

        previsoes_hibrido = np.zeros((10,20))
        rmse_hibrido = np.zeros((1,20))
        rmse_val_arima = np.zeros((1, 20))
        rmse_val_hibrido = np.zeros((1, 20))

        # iteração para realizar o procedimento para cada idade
        for idade, i in zip(idades, range(0,len(idades)+1)):
            lnmx = lnmx_series[idade]
            treino_i = treino[idade]
            teste_i = teste[idade]
            

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

            # previsao de validacao 
            arima_predict_val = arima.predict_in_sample()[-10:]
            rmse_val_arima[0,i] = MSE(treino_i.values[-10:], arima_predict_val, squared=False)
            
            # previsao arima - 21 períodos porque quando fizer a separação dos dados, vamos perder a última obs.
            arima_predict = arima.predict(n_periods=10, return_conf_int=False)
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

            predictions = np.zeros(10)
            val_predictions = np.zeros(arima_predict_val.size)
            for o in range(10):
                print('-----------------')
                print('HORIZONTE:,', o)
                print('-----------------')

                n_beats_model_error = treinar_nbeats(X_train_error, y_train_error[:, o], X_val_error, y_val_error[:, o], 10, execucao, idade, o)
                
                # validation pred
                X_val_pred = X_error[-1].reshape(1, 2)
                val_predictions[o] = prev_n_beats(n_beats_model_error, X_val_pred).ravel()

                ## Predictions
                X_test_pred = arima_error_norm[-2:].reshape(1, 2)
                n_beats_predict_error = prev_n_beats(n_beats_model_error, X_test_pred).ravel()
                predictions[o] = n_beats_predict_error

                # liberar a memória                
                del n_beats_model_error
                K.clear_session()
                gc.collect()
            
                        
            # Desnormalizar a série
            # val
            n_beats_val_error = desnormalizar(val_predictions, arima_error_train_test).values.ravel()
            z_val = arima_predict_val + n_beats_val_error
            rmse_val_hibrido[0, i] = MSE(treino_i.values[-10:], z_val, squared=False)

            # test
            n_beats_predict_error = desnormalizar(predictions, arima_error_train_test)
            n_beats_predict_error.index = teste_i.index 

            #Criando o sistema híbrido 
            z = arima_predict.loc[:2018]  + n_beats_predict_error[0]
            previsoes_hibrido[:,i] = z.values
            
            # RMSE
            rmse_hibrido[0,i] = MSE(teste_i, z, squared=False)
            print('-----------------------------------------------')
            print(f'IDADE: {idade}')
            print('RMSE para o Modelo n-beats direct = ', rmse_hibrido[0, i])
            print('-----------------------------------------------')


        # Salvando os resultados
        previsoes_hibrido = pd.DataFrame(previsoes_hibrido, columns = idades)
        rmse_hibrido = pd.DataFrame(rmse_hibrido, columns = idades)
        rmse_val_arima = pd.DataFrame(rmse_val_arima, columns=idades)
        rmse_val_hibrido = pd.DataFrame(rmse_val_hibrido, columns=idades)

        # Exportando os resultados para .csv
        previsoes_hibrido.to_csv(f'Resultados - hibrido - n-beats - Artigo - direct/previsoes_hibrido_{execucao}.csv', index=None, header=True, encoding = 'utf-8')
        rmse_hibrido.to_csv(f'Resultados - hibrido - n-beats - Artigo - direct/rmse_hibrido_{execucao}.csv', index=None, header=True, encoding = 'utf-8')
        rmse_val_arima.to_csv(f'Resultados - hibrido - n-beats - Artigo - direct/rmse_val_arima_{execucao}.csv', index=None, header=True, encoding='utf-8')
        rmse_val_hibrido.to_csv(f'Resultados - hibrido - n-beats - Artigo - direct/rmse_val_hibrido_{execucao}.csv', index=None, header=True, encoding='utf-8')


if __name__ == '__main__':
    print('Iniciando o script Hibrido N-Beats direct')
    main()
    print('FIM!')



