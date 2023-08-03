from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras import Model

def model_LSTM_1(input_shape):
    input_ = Input(shape=input_shape)

    lstm = LSTM(32, input_shape=input_shape, activation='tanh', return_sequences=True)(input_)
    lstm = Dropout(0.3)(lstm)
    lstm = LSTM(64)(lstm)
    lstm = Dropout(0.3)(lstm)

    DNN = Dense(64, activation='relu')(lstm)
    DNN = Dropout(0.3)(DNN)
    DNN = Dense(64, activation='relu')(DNN)
    DNN = Dropout(0.3)(DNN)
    DNN = Dense(10, activation='softmax')(DNN)

    model = Model(inputs=[input_], outputs=[DNN])

    return model


