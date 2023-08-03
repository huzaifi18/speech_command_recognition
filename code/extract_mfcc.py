import os
import librosa, librosa.display
import numpy as np
import pickle as pk
import pandas as pd
import warnings

import parameter as param

warnings.filterwarnings('ignore')


def extract_mfcc(n_mfcc, return_df):
    """Extract MFCC feature from list of dataset.

    Parameters
    ----------
    n_mfcc : int
        Number of MFCC's Coefficient
    return_df : bool


    Returns
    -------
    list
        list of extracted feature
        list of label
    """

    mfccs = []  # empty list for extracted feature
    all_label = []  # empty list for label
    print("Extracting Feature")
    for label in param.labels:
        print(label)
        waves = [f for f in os.listdir(param.MAIN_PATH + '/' + label) if
                 f.endswith('.wav')]  # initialize list of dataset
        for wav in waves:
            samples, sample_rate = librosa.load(param.MAIN_PATH + '/' + label + '/' + wav,
                                                sr=param.sr)  # load audio data
            if param.IDN_DATASET:
                samples = samples[:10000]
                if len(samples) == 10000:
                    mfcc = librosa.feature.mfcc(y=samples,
                                                sr=8000,
                                                n_mfcc=param.n_mfcc,
                                                hop_length=512,
                                                n_fft=param.n_fft
                                                )
                    mfccs.append(mfcc.T)
                    all_label.append(label)

            else:
                if len(samples) == 8000:
                    mfcc = librosa.feature.mfcc(y=samples,
                                                sr=8000,
                                                n_mfcc=param.n_mfcc,
                                                hop_length=512,
                                                n_fft=param.n_fft
                                                )
                    mfccs.append(mfcc.T)  # add extracted feature into empty list
                    all_label.append(label)  # add label into empty label list

    if return_df:
        label_entry = np.zeros(1)  # initialize empty array for dataframe label
        df_entry = np.zeros(shape=(1, n_mfcc))  # initialize empty array for extracted feature dataframe
        print("\nProcessing Dataframe")
        for label in param.labels:
            print(label)
            for mf in mfccs:
                df_entry = np.append(df_entry, mf, axis=0)  # stacking feature
                df_entry = df_entry[1:]  # slice first row (bcs fist row's value is zero)

                lb_ = [label] * len(mf)  # multiply label based on total entire frame extracted feature
                label_entry = np.append(label_entry, lb_, axis=0)  # stacking label
                label_entry = label_entry[1:]
        df = pd.DataFrame(df_entry)  # convert to dataframe
        df["label"] = label_entry  # initialize "label" column
        df.to_csv(param.CSV_MFCC_OUTPUT, index=False)

    with open(param.MFCC_PATH_OUTPUT, 'wb') as f:
        pk.dump(mfccs, f)

    with open(param.LABEL_MFCC_PATH_OUTPUT, 'wb') as f:
        pk.dump(all_label, f)


if __name__ == "__main__":
    extract_mfcc(n_mfcc=param.n_mfcc,
                 return_df=param.return_df)
