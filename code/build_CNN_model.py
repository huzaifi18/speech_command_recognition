from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras import Model


def model_P5(input_shape):
    input_ = Input(shape=input_shape)
    CNN = Conv1D(16, 3, padding='same', activation='relu')(input_)
    CNN = MaxPooling1D(3)(CNN)
    CNN = Dropout(0.2)(CNN)

    CNN = Conv1D(32, 3, padding='same', activation='relu')(CNN)
    CNN = MaxPooling1D(3)(CNN)
    CNN = Dropout(0.2)(CNN)

    CNN = Conv1D(64, 3, padding='same', activation='relu')(CNN)
    CNN = Dropout(0.2)(CNN)

    CNN = Conv1D(128, 3, padding='same', activation='relu')(CNN)
    CNN = Dropout(0.2)(CNN)

    flat = Flatten()(CNN)

    DNN = Dense(128, activation='relu')(flat)
    DNN = Dense(64, activation='relu')(DNN)
    DNN = Dense(64, activation='relu')(DNN)
    DNN = Dense(11, activation='softmax')(DNN)

    model = Model(inputs=[input_], outputs=[DNN])

    return model


def model_P6(input_shape):
    input_ = Input(shape=input_shape)
    CNN = Conv1D(8, 3, padding='same', activation='relu')(input_)
    CNN = Conv1D(16, 3, padding='same', activation='relu')(CNN)
    CNN = MaxPooling1D(3)(CNN)
    CNN = Dropout(0.2)(CNN)

    CNN = Conv1D(32, 3, padding='same', activation='relu')(CNN)
    CNN = MaxPooling1D(3)(CNN)
    CNN = Dropout(0.2)(CNN)

    CNN = Conv1D(64, 3, padding='same', activation='relu')(CNN)
    CNN = Dropout(0.2)(CNN)

    CNN = Conv1D(128, 3, padding='same', activation='relu')(CNN)
    CNN = Dropout(0.2)(CNN)

    flat = Flatten()(CNN)

    DNN = Dense(128, activation='relu')(flat)
    DNN = Dropout(0.2)(DNN)
    DNN = Dense(128, activation='relu')(DNN)
    DNN = Dropout(0.2)(DNN)
    DNN = Dense(128, activation='relu')(DNN)
    DNN = Dropout(0.2)(DNN)
    DNN = Dense(11, activation='softmax')(DNN)

    model = Model(inputs=[input_], outputs=[DNN])

    return model

def model_P7(input_shape):
    input_ = Input(shape=input_shape)
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

    flat = Flatten()(CNN)

    DNN = Dense(64, activation='relu')(flat)
    DNN = BatchNormalization()(DNN)
    DNN = Dropout(0.2)(DNN)
    DNN = Dense(64, activation='relu')(DNN)
    DNN = Dropout(0.2)(DNN)
    DNN = Dense(64, activation='relu')(DNN)
    DNN = Dropout(0.2)(DNN)
    DNN = Dense(11, activation='softmax')(DNN)

    model = Model(inputs=[input_], outputs=[DNN])

    return model
