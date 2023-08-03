from tensorflow.keras.layers import Input, Dense, GRU, Dropout
from tensorflow.keras import Model

def model_GRU_1(input_shape):
    input_ = Input(shape=input_shape)

    gru = GRU(32, input_shape=input_shape, activation='tanh', return_sequences=True)(input_)
    gru = Dropout(0.3)(gru)
    gru = GRU(64)(gru)
    gru = Dropout(0.3)(gru)

    DNN = Dense(64, activation='relu')(gru)
    DNN = Dropout(0.3)(DNN)
    DNN = Dense(64, activation='relu')(DNN)
    DNN = Dropout(0.3)(DNN)
    DNN = Dense(10, activation='softmax')(DNN)

    model = Model(inputs=[input_], outputs=[DNN])

    return model


