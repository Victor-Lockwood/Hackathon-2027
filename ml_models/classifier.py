import pandas as pd
import glob
import os
import os.path
import pickle
import itertools
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import csv

def main():
    model_path = "model.pkl"

    load_data()
    rf = None
    if os.path.isfile(model_path):
        rf = pickle.load(open(model_path, 'rb'))
    else:
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                           random_state=104,
                                           test_size=0.25,
                                           shuffle=True)

        rf = random_forest(X_train, y_train, X_test, y_test)
        pickle.dump(rf, open(model_path, 'wb'))



def random_forest(X_train, y_train, X_test, y_test):
    print("\n---- RANDOM FOREST TRAINING ----")
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    cr = classification_report(y_test, y_pred)
    print(cr)
    print("---- END RANDOM FOREST TRAINING ----")
    return rf

def load_data(window_size=150):
    # path = '../data/XDMyo-Dataset-main/A/csv'
    path = '../data/001/001'
    all_csvs = glob.glob(os.path.join(path, "*Myo_EMG.csv"))

    path2 = '../data/002/002'
    all_csvs.extend(glob.glob(os.path.join(path2, "*Myo_EMG.csv")))

    labels = []

    for name in all_csvs:
        with open(name) as f:
            label = next(itertools.islice(csv.reader(f), 1, None))
            label = label[0].replace("# label: ", '')
            labels.append(label)

    dataframes = []

    for idx, filename in enumerate(all_csvs):
        df = pd.read_csv(filename, index_col=None, header=0, comment='#')
        df['label'] = labels[idx]
        dataframes.append(df)

    myo_data = pd.concat(dataframes, axis=0, ignore_index=True)

    step = 50
    windows = [myo_data[i:i + window_size] for i in range(0, len(myo_data) - window_size + 1, step)]

    for window in windows:
        window['EMG_1'] = butter_bandstop_filter(window['EMG_1'])
        window['EMG_2'] = butter_bandstop_filter(window['EMG_2'])
        window['EMG_3'] = butter_bandstop_filter(window['EMG_3'])
        window['EMG_4'] = butter_bandstop_filter(window['EMG_4'])
        window['EMG_5'] = butter_bandstop_filter(window['EMG_5'])
        window['EMG_6'] = butter_bandstop_filter(window['EMG_6'])
        window['EMG_7'] = butter_bandstop_filter(window['EMG_7'])
        window['EMG_8'] = butter_bandstop_filter(window['EMG_8'])

    myo_data = pd.concat(windows, axis=0, ignore_index=True)

    X = myo_data.drop(['timestamp', 'label'], axis=1)
    y = myo_data['label']

    return X.values, y.values

def sim_continuous(rf=None, window_size=150):
    path = '../data/XDMyo-Dataset-main/A/csv/s2.csv'

    with open(path) as csvfile:
        reader = csv.reader(csvfile)  # change contents to floats
        idx = 0

        window = []
        for line in reader:
            idx += 1
            if idx == 1: continue

            if int(line[1]) == -1: continue

            window.append(line[1:10])
            # print(line[2:10])

            if len(window) % window_size == 0:
                arr = np.asarray(window)
                arr = arr.astype(int)
                arr = np.transpose(arr)

                y = arr[0]
                arr = np.delete(arr, 0, axis=0)
                arr = np.apply_along_axis(butter_bandstop_filter, 1, arr)

                arr = np.transpose(arr)

                # print(rf.predict(arr))

                window = []




def butter_bandstop_filter(data, order=4):
    lowcut = 55
    highcut = 65
    fs = 1000

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return filtfilt(b, a, data)


if __name__ == "__main__":
    main()