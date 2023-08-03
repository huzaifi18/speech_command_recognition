import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import numpy as np
import pandas as pd


def plot_accuracy(history, pth, title):
    # Plot model loss
    plt.figure(figsize=(10, 10))
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    plt.legend(loc="lower right", fontsize=18)
    plt.savefig(pth+".png", dpi=300)


def plot_confusion_matrix(y_true, y_pred, title, pth, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=18)
    plt.yticks(tick_marks, classes, fontsize=18)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=18)
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()
    plt.savefig(pth+".png", dpi=300)

def clf_report(y_true, y_pred, classes, pth):
    cr = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    df = pd.DataFrame(cr).T
    df.index.name = "idx"
    df.to_csv(pth+".csv")
