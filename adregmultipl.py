#Bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error



#Conectando base de dados
df = pd.read_csv('advertising.csv')
df = df.drop(['Unnamed: 0'], axis=1)
print(df.describe())

#DataFrame Variáveis independentes
df_nosales = df.drop(['newspaper','sales'], axis=1)
print('DataFrame var indpend: ',df_nosales)
 
#Modelo estatístico e métricas de precisão (2 Var independ.)
#Teste de Multicolinearidade e Dimensionalidade 
#Correlação das variáveis independentes
print(df_nosales.corr()) #Multicolinearidade -------- Não há multicolinearidade.


#Validação Cruzada
x_train = df_nosales
y_train = df['sales']


num_iterations = 4
quartil_size = len(x_train) // num_iterations # = 50
partitions = [x_train[i:i+quartil_size] for i in range(0, len(x_train), quartil_size)]
target_partitions = [y_train[i:i+quartil_size] for i in range(0, len(y_train), quartil_size)] 

mse_scoretrain ,mse_scoretest = [], [] #Cria dicionários vazios

for i in range(num_iterations): #Cria um laço que se repete 4 vezes.
    
    x_test, y_test = partitions[i], target_partitions[i] #Chama X(partitions[i]) e Y(target_predictions[1]) de variáveis de teste.
    x_train = np.vstack(partitions[:i] + partitions[i+1:])
    y_train = np.concatenate(target_partitions[:i] + target_partitions[i+1:]) #Validação vai até aqui.
    
    #Construção do modelo de regressão linear
    modelo = LinearRegression()
    modelo.fit(x_train, y_train)
    y_predtrain = modelo.predict(x_train) #Previsão do treino
    
    y_pred = modelo.predict(x_test) #Previsão do Teste 
    
    #Erros Quadráticos Médios
    msetrain = mean_squared_error(y_train, y_predtrain) #Erro Quadrático médio do Treino
    msetest = mean_squared_error(y_test, y_pred) #Erro Quadrático médio do Teste
    mse_scoretrain.append(msetrain)
    mse_scoretest.append(msetest)
     
    #Modelo de Regressão pelo Métodos OLS ----- Usado para medir precisão do Modelo de Regressão Linear Múltipla.
    X = x_train
    y = y_train
    X2 = sm.add_constant(X) #Adiciona costante ao modelo
    est = sm.OLS(y, X2) #Criando um mo delo
    est2 = est.fit() #Treinando o modelo estatístico
    print(est2.summary()) #Sumário com estatísticas descritivas
    print("O modelo é: Vendas = {:.5} + {:.5}*TV + {:.5}*radio".format(modelo.intercept_, modelo.coef_[0], modelo.coef_[1]))


print('Erros Quadrados de Treino: ', mse_scoretrain)
print('Erros Quadrados de Teste: ', mse_scoretest)
mse_scoretrain = pd.DataFrame(mse_scoretrain)
mse_scoretest = pd.DataFrame(mse_scoretest)

mean_msetrain = np.mean(mse_scoretrain)
mean_msetest = np.mean(mse_scoretest)
print('Média dos Erros Quadrados de Treino: ', mean_msetrain)
print('Média dos Erros Quadrados de Teste: ', mean_msetest)
variance = mse_scoretrain - mse_scoretest
print('Variância: ', variance.mean())
#Possuem Baixa Variância, além disso, Erros(Bias) baixo nos dois modelo. Logo, o modelo é ótimo.

print('Previsão de Vendas: ', y_pred)

x = range(0, 30) #Argumento de períodos

#Conjuntos
y_train = pd.DataFrame(y_train) #Treino
y_train30 = y_train.head(30)

y_test = pd.DataFrame(y_test)#Teste
y_test30 = y_test.head(30)
y_test30.reset_index()

#Predições
y_predtrain = pd.DataFrame(y_predtrain)
y_predtrain30 = y_predtrain.head(30)

y_pred = pd.DataFrame(y_pred)
y_pred30 = y_pred.head(30)


#Gráficos
#Gráfico Treino
plt.figure(figsize = (10, 5))
plt.plot(y_predtrain30, c='orange')
plt.plot(x, y_train30, color='navy')
plt.scatter(x, y_train30, color='navy', marker='o')
plt.ylabel('Vendas (em Milhões de US$)')
plt.title('Predições(Treino) x Conjunto de Treino')
plt.show()

#Gráfico Teste
plt.figure(figsize = (10,5))
plt.plot(y_pred30, c='orange')
plt.plot(x, y_test30, color='red')
plt.scatter(x, y_test30, color='orange', marker='o')
plt.title('Predições(Teste) x Conjunto de Teste')
plt.ylabel('Vendas (em Milhões de US$)')
plt.show()


