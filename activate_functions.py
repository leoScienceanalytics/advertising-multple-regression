from neuralnetwork_regress import preprocessing, rede_neural, grafico


file = 'advertising.csv'

X_train, x_test, y_train, y_test = preprocessing(file)
prev = rede_neural(X_train, y_train)
print(prev)

graph = grafico(prev[:30])
print(graph)