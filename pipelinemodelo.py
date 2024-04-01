from neuralnetwork_regress import preprocessing, rede_neural, grafico_treino

file = 'advertising.csv'

X_train, x_test, y_train, y_test = preprocessing(file) #Processamento e Feature engineering
prev, model, prev_test = rede_neural(X_train, y_train, x_test) #Construção da Rede e previsão
print(prev)

graph = grafico_treino(prev, y_train, prev_test, y_test) #Gráficos
print(graph)