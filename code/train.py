import json
import tensorflow as tf
import pickle
import numpy as np
import os
import random as rn

from numpy.random import seed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import train_param as param
import utils
import build_CNN_model

SEED = 1807
tf.keras.utils.set_random_seed(SEED)
# tf.config.experimental.enable_op_determinism()
os.environ['TF_CUDNN_DETERMINISTIC'] = "True"
os.environ["TF_DETERMINISTIC_OPS"] = "True"
os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "True"
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)

with open(param.dataset, 'rb') as f:
    X = pickle.load(f)
with open(param.label, 'rb') as fl:
    label = pickle.load(fl)

X = np.array(X)
LE = LabelEncoder()
y = LE.fit_transform(label)
classes = list(LE.classes_)

acc_per_fold = []
loss_per_fold = []
kfold = StratifiedKFold(n_splits=param.num_folds, shuffle=True, random_state=SEED)
# X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=param.test_size, random_state=42)

# print(X_train.shape, X_val.shape)

input_shape = (X.shape[1], X.shape[2])
print(input_shape)

i = 1
for train, test in kfold.split(X, y):
    print(f"Fold {i}")
    model = build_CNN_model.model_P7(input_shape)

    optim = Adam(learning_rate=param.lr)
    model.compile(optimizer=optim,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    print("Saving model structure")
    plot_model(model, to_file=param.plot_model_output, show_shapes=True)
    print("Done saving model structure")
    history = model.fit(X[train], y[train],
                        validation_data=(X[test], y[test]),
                        batch_size=param.batch_size,
                        epochs=param.epochs)

    print("Saving Model")
    model.save(param.save_model_dir+f"_k{i}"+".h5")

    y_pred = model.predict(X[test])
    y_pred = np.array(list(map(lambda x: np.argmax(x), y_pred)))

    print("Saving Accuracy Plot")
    utils.plot_accuracy(history, param.plot_accuracy_output+f"_k{i}", param.acc_plot_title)
    print("Saving Confusion Matrix")
    utils.plot_confusion_matrix(y[test], y_pred, param.cm_plot_title, param.plot_cm_output+f"_k{i}", classes)
    print("Saving Classification Report")
    utils.clf_report(y[test], y_pred, classes, param.csv_report_output+f"_k{i}")

    train_acc = history.history["accuracy"][-1]*100
    val_acc = history.history["val_accuracy"][-1]*100
    with open(param.txt_output+f"_k{i}"+".txt", 'w') as f:
        f.write("Train Accuracy: ")
        f.write(json.dumps(train_acc))
        f.write('\nVal Accuracy: ')
        f.write(json.dumps(val_acc))

    with open(param.history_output+f"_k{i}"+".json", 'w') as f:
        f.write(json.dumps(history.history))
    i = i + 1