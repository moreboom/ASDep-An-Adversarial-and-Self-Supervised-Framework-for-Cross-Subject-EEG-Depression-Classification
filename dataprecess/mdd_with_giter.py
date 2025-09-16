import os
import mne
import numpy as np
from sklearn.preprocessing import StandardScaler

def edf_to_numpy(file_path, selected_channels, window_size_sec=5, overlap=0.5):
    """
     EDF to NumPy and preprocess
     return shape (sample, channel, step)。
    """
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.pick_channels(selected_channels)
    #filter
    raw.filter(1, 40, fir_design='firwin')

    data = raw.get_data()
    sfreq = raw.info['sfreq']
    window_size = int(window_size_sec * sfreq)
    step_size = int(window_size * (1 - overlap))
    n_channels, n_samples = data.shape
    n_segments = (n_samples - window_size) // step_size + 1

    segments = np.zeros((n_segments, n_channels, window_size))
    for i in range(n_segments):
        start = i * step_size
        end = start + window_size
        segments[i] = data[:, start:end]

    # Z-score
    scaler = StandardScaler()
    segments = np.array([scaler.fit_transform(seg.T).T for seg in segments])
    return segments

if __name__ == "__main__":
    #input_dir
    in_dir = "19channel"

    #output_dir
    #####################################
    out_dir = "19channel/19"
    ####################################


    os.makedirs(out_dir, exist_ok=True)



    # channel select
    #it was named according to dataset of MDD
    ##################################
    selected_channels = [
        'EEG Fp1-LE', 'EEG Fp2-LE',
        'EEG F3-LE', 'EEG F4-LE',
        'EEG C3-LE', 'EEG C4-LE',
        'EEG P3-LE', 'EEG P4-LE',
        'EEG O1-LE', 'EEG O2-LE',
        'EEG F7-LE', 'EEG F8-LE',
        'EEG T3-LE', 'EEG T4-LE',
        'EEG T5-LE', 'EEG T6-LE',
        'EEG Fz-LE', 'EEG Cz-LE',
        'EEG Pz-LE'
    ]

    ####################################






########################################################
    # process H S1 EC to H S27 EC，If use other document, please change
    for idx in range(1, 31):
        # name format
        fname = f"H S{idx} EC.edf"
        #########################################





        fpath = os.path.join(in_dir, fname)
        if not os.path.isfile(fpath):
            print(f"[skip] can't find document：{fpath}")
            continue

        print(f"[process] {fpath}")
        segments = edf_to_numpy(fpath, selected_channels,
                                window_size_sec=5, overlap=0.5)###here to change the length of step



        # save .npy，ex: "H_S1_EC.npy"
        ##################################
        out_name = f"H_S{idx}_EC.npy"
        ####################################




        out_path = os.path.join(out_dir, out_name)
        np.save(out_path, segments)
        print(f"→ saved {out_path}，shape = {segments.shape}")

    print("finish！")
