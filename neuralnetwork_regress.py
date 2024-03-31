def preprocessing(file):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    dados = pd.read_csv(file)
    target = dados['sales'].values
    features = dados.drop(['sales'], axis=1).values
    X_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3)
    return X_train, x_test, y_train, y_test


def rede_neural(X_train, y_train, X_test):
    import pandas as pd
    from keras.models import Sequential 
    from keras.layers import Dense, Dropout 
        
    #Parâmetros
    #Camadas
    neuronio_entrada = float(input('Camada de entrada:'))
    neuronio_oculta_dense = float(input('Camada Oculta:'))        
    neuronio_oculta_dropout = float(input('Camada Oculta de perda:'))
    neuronio_saida = float(input('Camada de saída:'))    
    funcao_ativacao = str(input('Função de ativação:'))
    #Compile
    otimizador = str(input('Optimizer:'))
    loss = str(input('Perda:'))
    #Fit
    epoca = int(input('Épocas:'))
        
    model = Sequential()
    model.add(Dense(units=neuronio_entrada, activation=funcao_ativacao, input_shape =(4,)))
    model.add(Dense(units=neuronio_oculta_dense, activation=funcao_ativacao))
    model.add(Dropout(neuronio_oculta_dropout))
    model.add(Dense(units=neuronio_saida, activation=funcao_ativacao))
    model.compile(optimizer=otimizador, loss=loss, metrics=['mae'])
    model.summary()
    model.fit(X_train, y_train, epochs=epoca, batch_size=10, verbose=True, validation_data=(X_train, y_train))
    nova_prev = model.predict(X_train)
    prev = pd.DataFrame(nova_prev)
    
    prev_test = model.predict(X_test)
    prev_test = pd.DataFrame(prev_test)
    
    return prev, model, prev_test


def grafico_treino(prev, y_train, prev_test, y_test):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10,5))
    plt.plot(y_train[:30], color='Blue', label='Treinamento')
    plt.plot(prev[:30], color='Orange', label='Previsão do Treinamento')
    plt.legend(loc='upper left', fontsize= 10)
    plt.xlabel('Período')
    plt.ylabel('Valor')
    plt.show()
    
    
    plt.figure(figsize=(10,5))
    plt.plot(y_test[:30], color='Red', label='Teste')
    plt.plot(prev_test[:30], color='Green', label='Previsão do Teste')
    plt.legend(loc='upper left', fontsize= 10)
    plt.xlabel('Período')
    plt.ylabel('Valor')
    plt.show()
    
    
file = 'advertising.csv'

X_train, x_test, y_train, y_test = preprocessing(file) #Processamento e Feature engineering
prev, model, prev_test = rede_neural(X_train, y_train, x_test) #Construção da Rede e previsão
print(prev)

graph = grafico_treino(prev, y_train, prev_test, y_test) #Gráficos
print(graph)