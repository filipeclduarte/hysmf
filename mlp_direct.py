## MLP para Previsão de Séries Temporais
# Importação das bibliotecas
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as MSE
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import math
import itertools
import optunity
import optunity.metrics

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

def treinar_mlp(x_train, y_train, x_val, y_val, num_exec, execucao, idade, o):
    
    neuronios =  [2, 5, 10, 15, 20, 100]
    func_activation =  ['tanh', 'relu']
    alg_treinamento = 'adam' # ['sgd', 'adam']
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
        print(f'Rodada {rodada}')

        for neuronio in neuronios:
            for f_act in func_activation:
                for iteracao in iteracoes:
                    for qtd_lag in range(1, x_train.shape[1] + 1):
                        mlp = MLPRegressor(hidden_layer_sizes=neuronio, 
                        activation=f_act, 
                        solver=alg_treinamento, 
                        max_iter=iteracao, 
                        learning_rate=learning_rate[0])
                        mlp.fit(x_train[:, -qtd_lag:], y_train)
                        predict_validation = mlp.predict(x_val[:, -qtd_lag:])
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

        # salvar o melhor modelo da rodada 
        print('Execução:', execucao, 'idade:', idade, 'horizonte:', o, 'rodada:', rodada, 'melhor configuração neuronios:', melhor_neuronio,
        'qtd_lag:', qtd_lags_sel, 'RMSE:', melhor_mse)
        with open(f'Resultados - MLP - Artigo - Horizontes/exec_{execucao}_model_mlp_horizonte_{o}_rodada_{rodada}_idade_{idade}.sav','wb') as file:
            pickle.dump(melhor_modelo, file)

        
    # salvar o dataframe com todos os resultados
    resultados_df = pd.DataFrame({'rodada': rodadas_lista, 'neuronios': neuronios_lista, 'func_activation': func_ativacao_lista,'qtd_lag': qtd_lag_lista, 'iteracoes': iteracoes_lista, 'rmse': rmse_lista})
    resultados_df.to_csv(f'Resultados - MLP - Artigo - Horizontes/exec_{execucao}_mlp_idade_{idade}_horizonte_{o}_resultados_df.csv')
    # obter o melhor modelo
    melhor_mse_i = resultados_df['rmse'].argmin()
    melhor_rodada = int(resultados_df.iloc[melhor_mse_i]['rodada'])
    with open(f'Resultados - MLP - Artigo - Horizontes/exec_{execucao}_model_mlp_horizonte_{o}_rodada_{melhor_rodada}_idade_{idade}.sav', 'rb') as file:
        melhor_modelo = pickle.load(file)
    # obter a melhor qtd de lags
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
        print(f'Execução {execucao}')
        np.random.seed(execucao + 5)
        previsoes_MLP = np.zeros((10,20))
        rmse_MLP = np.zeros((1, 20))

        for idade, i in zip(idades, range(0,len(idades)+1)):
            lnmx = lnmx_series[idade]
            treino_i = treino[idade]
            teste_i = teste[idade]

            # normalizando a série para MLP
            lnmx_norm = normalizar_serie(lnmx.values)
            lnmx_train_norm = lnmx_norm[:-10]

            # Separando os dados em X, y e utilizando 2 lags 
            X, y = split_sequence(lnmx_train_norm, 2, 10)
            X_train, y_train, X_val, y_val = divisao_dados_temporais(X, y, perc_treino = 0.80)

            # realizar as previsões com 10 modelos, sendo um para cada step
            predictions = np.zeros(10)
            for o in range(10):
                print('-----------------')
                print('HORIZONTE:,', o)
                print('-----------------')
                MLP_model, lag_sel = treinar_mlp(X_train, y_train[:,o], X_val, y_val[:,o], 10, execucao, idade, o)
                
                X_test_pred = lnmx_train_norm[-lag_sel:].reshape(1, lag_sel)
                MLP_predict = MLP_model.predict(X_test_pred)
                predictions[o] = MLP_predict
            
            # desnormalizar as previsões     
            MLP_predict = desnormalizar(predictions, lnmx)

            # Salvando na matrix previsoes_MLP    
            previsoes_MLP[:,i] = MLP_predict.values.reshape(10,)

            # transformar em série para realizar o plot
            MLP_predict.index = teste_i.index

            # RMSE
            rmse_MLP[0,i] = np.sqrt(MSE(teste_i, MLP_predict))
            print('-----------------------------------------------')
            print('RMSE para MLP = ', np.sqrt(MSE(teste_i, MLP_predict)))
            print('-----------------------------------------------')
        # Salvando os resultados
        previsoes_MLP = pd.DataFrame(previsoes_MLP, columns = idades)
        rmse_MLP = pd.DataFrame(rmse_MLP, columns = idades)

        # Exportando os resultados para .csv
        previsoes_MLP.to_csv('Resultados - MLP - Artigo - Horizontes/previsoes_MLP_serie.csv', index=None, header=True, encoding = 'utf-8')
        rmse_MLP.to_csv('Resultados - MLP - Artigo - Horizontes/rmse_MLP_serie.csv', index=None, header=True, encoding = 'utf-8')


if __name__ == '__main__':
    print('Iniciando o script MLP')
    main()
    print('FIM!')

