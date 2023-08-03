import os
import librosa, librosa.display
import numpy as np
import pickle as pk
import pandas as pd
import warnings
from spafe.features.pncc import pncc
import parameter as param

warnings.filterwarnings('ignore')


def extract_pncc(path, n_pncc, labels, return_df=True):
    """Extract pncc's feature from list of dataset.

    Parameters
    ----------
    path : str
        Dataset directory
    n_pncc : int
        Number of pncc's coefficient
    labels : list
        A list of label
    return_df : bool
        want to create DataFrame or not

    Returns
    -------
    list
        list of extracted feature
        list of label
    pd.DataFrame
        Pandas Dataframe with number of window as row and feature and label as column
    """

    pnccs = []  # empty list for extracted feature
    all_label = []  # empty list for label

    print("Extracting Feature")
    for label in labels:
        print(label)
        waves = [f for f in os.listdir(path + '/' + label) if f.endswith('.wav')]  # initialize list of dataset
        for wav in waves:
            samples, sample_rate = librosa.load(path + '/' + label + '/' + wav, sr=8000)  # load audio data

            if param.IDN_DATASET:
                samples = samples[:10000]
                if len(samples) == 10000:
                    pncc_s = pncc(sig=samples,
                                  fs=param.sr,
                                  num_ceps=n_pncc,
                                  low_freq=0,
                                  high_freq=None,
                                  nfft=param.n_fft)
                    pnccs.append(pncc_s)  # add extracted feature into empty list
                    all_label.append(label)  # add label into empty label list
            else:
                if len(samples) == 8000:
                    pncc_s = pncc(sig=samples,
                                  fs=param.sr,
                                  num_ceps=n_pncc,
                                  low_freq=0,
                                  high_freq=None,
                                  nfft=param.n_fft)
                    if not np.any(np.isnan(pncc_s)):
                        pnccs.append(pncc_s)  # add extracted feature into empty list
                        all_label.append(label)  # add label into empty label list

    if return_df:
        label_entry = np.zeros(1)  # initialize empty array for dataframe label
        df_entry = np.zeros(shape=(1, n_pncc))  # initilazie empty array for extracted feature dataframe
        print("\nProcessing Dataframe")
        for label in labels:
            print(label)
            for mf in pnccs:
                df_entry = np.append(df_entry, mf, axis=0)  # stacking feature
                df_entry = df_entry[1:]  # slice first row (bcs fist row's value is zero)

                lb_ = [label] * len(mf)  # multiply label based on total entire frame extracted feature
                label_entry = np.append(label_entry, lb_, axis=0)  # stacking label
                label_entry = label_entry[1:]
        df = pd.DataFrame(df_entry)  # convert to dataframe
        df["label"] = label_entry  # initialize "label" column
        df.to_csv(param.CSV_PNCC_OUTPUT, index=False)

    with open(param.PNCC_PATH_OUTPUT, 'wb') as f:
        pk.dump(pnccs, f)

    with open(param.LABEL_PNCC_PATH_OUTPUT, 'wb') as f:
        pk.dump(all_label, f)

    return pnccs, all_label

if __name__ == "__main__":
    extract_pncc(path=param.MAIN_PATH,
                 n_pncc=param.n_pncc,
                 labels=param.labels,
                 return_df=param.return_df)