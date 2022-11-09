## Sistemas Híbridos para Previsão de Séries Temporais

# Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
from pmdarima.arima import auto_arima
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as MSE

import optunity.metrics
import gc
import pickle

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

def treinar_mlp(x_train, y_train, x_val, y_val, num_exec, execucao, idade):
    
    # neuronios =  [2, 3, 4, 5, 6, 7, 8, 9,10, 20, 30, 40, 50, 60, 70, 80, 90, 100]    
    neuronios = [2, 5, 10, 15, 20, 100]
    func_activation =  ['tanh', 'relu']
    alg_treinamento = 'adam'
    iteracoes = [1000] 
    learning_rate = ['adaptive']  #['constant', 'adaptive']
    qtd_lags_sel = x_train.shape[1]
    # best_result = np.Inf

    rodadas_lista = []
    neuronios_lista = []
    func_ativacao_lista = []
    # alg_treinamento_lista = []
    qtd_lag_lista = []
    iteracoes_lista = []
    rmse_lista = []


    for rodada in range(num_exec):
        melhor_mse = np.Inf
        melhor_neuronio = None
        melhor_modelo = None
        print(f'Rodada: {rodada}')

        for neuronio in neuronios:
            for f_act in func_activation:
                for iteracao in iteracoes:
                    for qtd_lag in range(1, x_train.shape[1]+1):
                        mlp = MLPRegressor(
                                            hidden_layer_sizes=neuronio, activation=f_act,
                                            solver=alg_treinamento, max_iter = iteracao, 
                                            learning_rate=learning_rate[0]
                                            )

                        mlp.fit(x_train[:,-qtd_lag:], y_train)
                        predict_validation = mlp.predict(x_val[:,-qtd_lag:])
                        novo_mse = np.sqrt(MSE(y_val, predict_validation))

                        rodadas_lista.append(rodada)
                        neuronios_lista.append(neuronio)
                        func_ativacao_lista.append(f_act)
                        qtd_lag_lista.append(qtd_lag)
                        iteracoes_lista.append(iteracao)
                        rmse_lista.append(novo_mse)

                        if novo_mse < melhor_mse:
                            melhor_mse = novo_mse
                            melhor_modelo = mlp
                            qtd_lags_sel = qtd_lag
                            melhor_neuronio = neuronio
                            # print('Execução:',execucao,'idade:', idade,'rodada:', rodada,'melhor configuração neuronios:', 
                            #         neuronio, 'qtd_lag:', qtd_lag, 'RMSE:', melhor_mse)
                            # with open(f'Resultados - Híbrido - ARIMA - MLP_exec_{execucao}/model_mlp_rodada_{rodada}_idade_{idade}.sav', 'wb') as file:
                            #     pickle.dump(mlp, file)
        # Salvar o melhor modelo da rodada (posso fazer dessa forma pq não preciso deletar os modelos)
        print('Execução:',execucao,'idade:', idade,'rodada:', rodada,'melhor configuração neuronios:', melhor_neuronio, 'qtd_lag:', qtd_lags_sel, 'RMSE:', melhor_mse)
        with open(f'Resultados - Híbrido - MLP_exec_{execucao}/model_mlp_rodada_{rodada}_idade_{idade}.sav', 'wb') as file:
            pickle.dump(melhor_modelo, file)

    # agora vamos salvar o dataframe com todos os resultados
    resultados_df = pd.DataFrame({'rodada': rodadas_lista, 'neuronios': neuronios_lista, 'func_activation': func_ativacao_lista,'qtd_lag': qtd_lag_lista, 'iteracoes': iteracoes_lista, 'rmse': rmse_lista})
    resultados_df.to_csv(f'Resultados - Híbrido - MLP_exec_{execucao}/mlp_idade_{idade}_resultados_df.csv')
    # obter o melhor modelo
    melhor_mse_i = resultados_df['rmse'].argmin()
    melhor_rodada = int(resultados_df.iloc[melhor_mse_i]['rodada'])
    with open(f'Resultados - Híbrido - MLP_exec_{execucao}/model_mlp_rodada_{melhor_rodada}_idade_{idade}.sav', 'rb') as file:
        melhor_modelo = pickle.load(file)
    # obter a qtd de lag adequada
    qtd_lags_sel = int(resultados_df.iloc[melhor_mse_i]['qtd_lag'])

    return melhor_modelo, qtd_lags_sel
        
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
        print(f'Execução: {execucao}')
        np.random.seed(execucao + 5)
        
        previsoes_arima = np.zeros((10,20)) 
        previsoes_error_mlp = np.zeros((10,20))
        previsoes_hibrido = np.zeros((10,20))
        rmse_arima = np.zeros((1,20))
        rmse_error_mlp = np.zeros((1,20))
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
            
            # treinar MLP no erro do arima
            mlp_model_error, lag_sel_error = treinar_mlp(X_train_error, y_train_error, X_val_error, y_val_error, 10, execucao, idade)
            
            # Realizar a previsão
            # mlp_predict_error = mlp_model_error.predict(X_test_error[-1, -lag_sel_error:].reshape(1,lag_sel_error))
            ## Testar com arima_erro_train_test_norm
            # X_test_pred = arima_error_train_test_norm[-(lag_sel_error+20):-20].reshape(1, lag_sel_error)
            X_test_pred = arima_error_norm[-lag_sel_error:].reshape(1, lag_sel_error)
            mlp_predict_error = mlp_model_error.predict(X_test_pred).reshape(10,)

            # transformar em série
            # mlp_predict_error = pd.Series(mlp_predict_error.reshape(10,), index = teste.index)

            # Desnormalizar a série
            mlp_predict_error = desnormalizar(mlp_predict_error, arima_error_train_test)
            
            # previsoes_error_mlp[:,i] = mlp_predict_error.values.reshape(1,-1)
            previsoes_error_mlp[:,i] = mlp_predict_error.values.reshape(10,)
            
            # mlp_predict_error = mlp_predict_error[0]
            mlp_predict_error.index = teste_i.index

            #Criando o sistema híbrido 
            # z = arima_predict.loc[:2018] + mlp_predict_error
            z = arima_predict.loc[:2018] + mlp_predict_error[0]
            # z = pd.Series(z, index = teste.index)
            
            # previsoes_hibrido[:,i] = z
            previsoes_hibrido[:,i] = z.values 
            
            # RMSE
            rmse_arima[0,i] = np.sqrt(MSE(teste_i, arima_predict.loc[:2018]))
            rmse_error_mlp[0,i] = np.sqrt(MSE(arima_error_predict, mlp_predict_error))
            rmse_hibrido[0,i] = np.sqrt(MSE(teste_i, z))
            print('-----------------------------------------------')
            print('Ln(mx) para a idade ', idade, 'RMSE - ARIMA = ',np.sqrt(MSE(teste_i, arima_predict.loc[:2018])))
            print('RMSE para o Modelo híbrido Z = ARIMA + MLP = ', np.sqrt(MSE(teste_i, z)))
            print('-----------------------------------------------')

        # Salvando os resultados
        #previsoes_arima = pd.DataFrame(previsoes_arima, columns= idades)
        previsoes_error_mlp = pd.DataFrame(previsoes_error_mlp, columns = idades)
        previsoes_hibrido = pd.DataFrame(previsoes_hibrido, columns = idades)
        #rmse_arima = pd.DataFrame(rmse_arima, columns = idades)
        rmse_error_mlp = pd.DataFrame(rmse_error_mlp, columns = idades)
        rmse_hibrido = pd.DataFrame(rmse_hibrido, columns = idades)

        # Exportando os resultados para .csv
        #previsoes_arima.to_csv('previsoes_arima.csv', index=None, header=True, encoding = 'utf-8')
        previsoes_error_mlp.to_csv(f'Resultados - Híbrido - MLP_exec_{execucao}/previsoes_error_mlp_exec_{execucao}.csv', index=None, header=True, encoding = 'utf-8')
        previsoes_hibrido.to_csv(f'Resultados - Híbrido - MLP_exec_{execucao}/previsoes_hibrido_exec_{execucao}.csv', index=None, header=True, encoding = 'utf-8')
        #rmse_arima.to_csv('rmse_arima.csv', index=None, header=True, encoding = 'utf-8')
        rmse_error_mlp.to_csv(f'Resultados - Híbrido - MLP_exec_{execucao}/rmse_error_mlp_exec_{execucao}.csv', index=None, header=True, encoding = 'utf-8')
        rmse_hibrido.to_csv(f'Resultados - Híbrido - MLP_exec_{execucao}/rmse_hibrido_exec_{execucao}.csv', index=None, header=True, encoding = 'utf-8')

if __name__ == '__main__':
    print('Iniciando o script Híbrido ARIMA-MLP')
    main()
    print('FIM!')