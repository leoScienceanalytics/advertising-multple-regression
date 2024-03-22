def preprocessing(file):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    dados = pd.read_csv(file)
    target = dados['sales'].values
    features = dados.drop(['sales'], axis=1).values
    X_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3)
    return X_train, x_test, y_train, y_test


def rede_neural(X_train, y_train):
    import pandas as pd
    from keras.models import Sequential 
    from keras.layers import Dense, Dropout 
        
    model = Sequential()
    model.add(Dense(units=3, activation='relu', input_shape =(4,)))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=1, activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.summary()
    model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=True, validation_data=(X_train, y_train))
    nova_prev = model.predict(X_train)
    prev = pd.DataFrame(nova_prev)
    return prev


def grafico(prev):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10,5))
    plt.plot(prev)
    plt.xlabel('Período')
    plt.ylabel('Valor')
    plt.show()