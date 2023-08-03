from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization, LSTM, concatenate
from tensorflow.keras import Model


def model_1(input_shape):
    input_ = Input(shape=input_shape)

    # CNN
    CNN = Conv1D(8, 5, padding='same', activation='relu')(input_)
    CNN = BatchNormalization()(CNN)
    CNN = Conv1D(16, 5, padding='same', activation='relu')(CNN)
    CNN = BatchNormalization()(CNN)
    CNN = MaxPooling1D(3)(CNN)
    CNN = Dropout(0.2)(CNN)

    CNN = Conv1D(32, 5, padding='same', activation='relu')(CNN)
    CNN = BatchNormalization()(CNN)
    CNN = MaxPooling1D(3)(CNN)
    CNN = Dropout(0.2)(CNN)

    CNN = Conv1D(64, 5, padding='same', activation='relu')(CNN)
    CNN = BatchNormalization()(CNN)
    CNN = Dropout(0.2)(CNN)

    CNN = Conv1D(128, 5, padding='same', activation='relu')(CNN)
    CNN = BatchNormalization()(CNN)
    CNN = Dropout(0.2)(CNN)
    CNN = Flatten()(CNN)

    # LSTM
    lstm = LSTM(64, input_shape=input_shape, activation='tanh', return_sequences=True)(input_)
    lstm = Dropout(0.3)(lstm)
    lstm = LSTM(64, activation='tanh')(lstm)
    lstm = Dropout(0.3)(lstm)

    concat = concatenate([CNN, lstm])

    DNN = Dense(128, activation='relu')(concat)
    DNN = BatchNormalization()(DNN)
    DNN = Dropout(0.2)(DNN)
    DNN = Dense(128, activation='relu')(DNN)
    DNN = Dropout(0.2)(DNN)
    DNN = Dense(128, activation='relu')(DNN)
    DNN = Dropout(0.2)(DNN)
    DNN = Dense(10, activation='softmax')(DNN)

    model = Model(inputs=[input_], outputs=[DNN])

    return model