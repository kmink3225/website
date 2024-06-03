# simple nueral network

def compute_simple_nn(y):
    x = np.linspace(0, 1, len(y))

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    model = Sequential([
        Dense(8, activation='relu', input_dim=1),
        Dense(8, activation='softplus'), #softplus
        Dense(2, activation='selu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    history = model.fit(x_norm, y_norm, epochs=500, verbose=0)
    y_pred = model.predict(x_norm)
    y_pred_denorm = y_pred * (y.max() - y.min()) + y.min()
    o_result = [item for sublist in y_pred_denorm for item in sublist]

    return o_result
