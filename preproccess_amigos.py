"""
AMIGOS Dataset - ECG & EDA Preprocessing + Feature Extraction
==============================================================

Select the main dataset folder containing the subject subfolders.

Expected structure:
    AMIGOS/
        Data_Preprocessed_P01/
            Data_Preprocessed_P01.mat
        Data_Preprocessed_P02/
            ...

Channels in joined_data (17 columns):
    0-13 : EEG
    14   : ECG
    15   : ECG2
    16   : GSR / EDA (raw ADC values)

Labels in labels_selfassessment (per trial):
    column 0 : Arousal
    column 1 : Valence

Output -> amigos_output/ folder in the same directory as the script:
    - amigos_features.csv   (features + signal quality for each trial)
    - plots/

"""
import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import neurokit2 as nk
import warnings
warnings.filterwarnings('ignore')
 
from tkinter import Tk, filedialog
 
# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
ECG_FS = 256   # Hz — ECG sampling rate
EDA_FS = 128   # Hz — EDA/GSR sampling rate
ECG_CH = 14    # ECG column index in joined_data
EDA_CH = 16    # EDA/GSR column index in joined_data
 
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "amigos_output")
 
# ─────────────────────────────────────────────────────────────────────────────
# LOAD .MAT
# ─────────────────────────────────────────────────────────────────────────────
 
def load_mat(filepath):
    try:
        return sio.loadmat(filepath, simplify_cells=True)
    except NotImplementedError:
        import h5py
        mat = {}
        with h5py.File(filepath, 'r') as f:
            for k in f.keys():
                mat[k] = f[k][()]
        return mat
 
def get_trials_and_labels(mat):
    trials_raw = mat['joined_data']
    labels_raw = mat['labels_selfassessment']
 
    trials   = []
    valences = []
    arousals = []
 
    for i in range(len(trials_raw)):
        label = np.array(labels_raw[i]).flatten()
        if label.size < 2:
            continue
        trials.append(np.array(trials_raw[i], dtype=float))
        arousals.append(float(label[0]))
        valences.append(float(label[1]))
 
    return trials, valences, arousals
 
# ─────────────────────────────────────────────────────────────────────────────
# FILTERS
# ─────────────────────────────────────────────────────────────────────────────
 
def filter_ecg(ecg, fs=ECG_FS):
    """Bandpass 0.5-40 Hz + 50 Hz notch filter."""
    nyq = fs / 2.0
    b, a = signal.butter(4, [0.5/nyq, 40.0/nyq], btype='band')
    ecg_bp = signal.filtfilt(b, a, ecg)
    b2, a2 = signal.iirnotch(50.0 / nyq, Q=30)
    return signal.filtfilt(b2, a2, ecg_bp)
 
def filter_eda(eda, fs=EDA_FS):
    """ADC normalization -> [0,1] + 5 Hz lowpass filter."""
    eda_min, eda_max = eda.min(), eda.max()
    if eda_max > eda_min:
        eda_norm = (eda - eda_min) / (eda_max - eda_min)
    else:
        eda_norm = eda.copy()
    nyq = fs / 2.0
    b, a = signal.butter(4, 5.0/nyq, btype='low')
    return signal.filtfilt(b, a, eda_norm)
 
# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL QUALITY — two separate indices, range [0-1]
# ─────────────────────────────────────────────────────────────────────────────
 
def _snr_estimate(sig, fs):
    f, pxx = signal.welch(sig, fs=fs, nperseg=min(256, len(sig) // 2))
    if len(f) < 2:
        return 0.0
    sp = np.sum(pxx[f < 10])
    tp = np.sum(pxx)
    return float(np.clip(sp / tp, 0, 1)) if tp > 0 else 0.0
 
def compute_ecg_quality(ecg_filt):
    """
    ECG quality index [0-1]:
      0.5 * spectral_SNR  +  0.5 * fraction_of_physiological_RR_intervals
    """
    ecg_snr = _snr_estimate(ecg_filt, ECG_FS)
    try:
        _, info  = nk.ecg_process(ecg_filt, sampling_rate=ECG_FS)
        rpeaks   = info["ECG_R_Peaks"]
        if len(rpeaks) >= 2:
            rr_ms    = np.diff(rpeaks) / ECG_FS * 1000.0
            rr_valid = np.sum((rr_ms >= 300) & (rr_ms <= 1500)) / len(rr_ms)
        else:
            rr_valid = 0.0
    except Exception:
        rr_valid = 0.0
    return round(float(np.clip(0.5 * ecg_snr + 0.5 * rr_valid, 0, 1)), 4)
 
def compute_eda_quality(eda_filt):
    """
    EDA quality index [0-1]:
      spectral_SNR * (0.5 + 0.5 * normalized_range)
    """
    eda_snr   = _snr_estimate(eda_filt, EDA_FS)
    eda_range = float(np.clip(np.ptp(eda_filt), 0, 1))
    return round(float(np.clip(eda_snr * (0.5 + 0.5 * eda_range), 0, 1)), 4)
 
# ─────────────────────────────────────────────────────────────────────────────
# ECG FEATURES
# ─────────────────────────────────────────────────────────────────────────────
 
def ecg_features(ecg_filt, fs=ECG_FS):
    try:
        _, info = nk.ecg_process(ecg_filt, sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 4:
            raise ValueError("Too few R-peaks detected")
        rr = np.diff(rpeaks) / fs * 1000.0  # ms
        hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=fs)
        lf = float(hrv_freq['HRV_LF'].values[0])
        hf = float(hrv_freq['HRV_HF'].values[0])
        return {
            'ecg_hr_mean':  60000.0 / np.mean(rr),
            'ecg_sdnn':     np.std(rr),
            'ecg_rmssd':    np.sqrt(np.mean(np.diff(rr)**2)),
            'ecg_pnn50':    np.sum(np.abs(np.diff(rr)) > 50) / len(rr) * 100,
            'ecg_mean_rr':  np.mean(rr),
            'ecg_lf_power': lf,
            'ecg_hf_power': hf,
            'ecg_lf_hf':    lf / hf if hf > 0 else np.nan,
        }
    except Exception as e:
        print(f"    ⚠ ecg_features failed: {e}")
        return {k: np.nan for k in
                ['ecg_hr_mean','ecg_sdnn','ecg_rmssd','ecg_pnn50',
                 'ecg_mean_rr','ecg_lf_power','ecg_hf_power','ecg_lf_hf']}
 
# ─────────────────────────────────────────────────────────────────────────────
# EDA FEATURES
# ─────────────────────────────────────────────────────────────────────────────
 
def eda_features(eda_filt, fs=EDA_FS):
    try:
        eda_signals, info = nk.eda_process(eda_filt, sampling_rate=fs)
 
        tonic  = eda_signals['EDA_Tonic'].values
        phasic = eda_signals['EDA_Phasic'].values
 
        # Safely retrieve SCR_Peaks (neurokit2 0.2.x returns numpy arrays)
        scr_idx = np.array([])
        if 'SCR_Peaks' in info and info['SCR_Peaks'] is not None:
            scr_idx = np.array(info['SCR_Peaks'])
        elif 'SCR_Onsets' in info and info['SCR_Onsets'] is not None:
            scr_idx = np.array(info['SCR_Onsets'])
 
        n_scr   = len(scr_idx)
        dur_min = len(eda_filt) / fs / 60.0
 
        return {
            'eda_mean':        float(np.mean(eda_filt)),
            'eda_std':         float(np.std(eda_filt)),
            'eda_scl_mean':    float(np.mean(tonic)),
            'eda_scl_std':     float(np.std(tonic)),
            'eda_scl_range':   float(np.max(tonic) - np.min(tonic)),
            'eda_phasic_mean': float(np.mean(phasic)),
            'eda_phasic_max':  float(np.max(phasic)),
            'eda_scr_count':   int(n_scr),
            'eda_scr_rate':    float(n_scr / dur_min) if dur_min > 0 else np.nan,
        }
 
    except Exception as e:
        print(f"    ⚠ eda_features failed: {e}")
        return {k: np.nan for k in
                ['eda_mean','eda_std','eda_scl_mean','eda_scl_std','eda_scl_range',
                 'eda_phasic_mean','eda_phasic_max','eda_scr_count','eda_scr_rate']}
 
# ─────────────────────────────────────────────────────────────────────────────
# PLOT raw vs filtered
# ─────────────────────────────────────────────────────────────────────────────
 
def save_plot(ecg_raw, ecg_filt, eda_raw, eda_filt, subj_name, trial_idx, plot_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle(f"{subj_name} – Trial {trial_idx + 1}", fontsize=13, fontweight='bold')
 
    t_ecg = np.arange(len(ecg_raw)) / ECG_FS
    t_eda = np.arange(len(eda_raw)) / EDA_FS
 
    axes[0, 0].plot(t_ecg, ecg_raw,  color='#e74c3c', lw=0.6)
    axes[0, 0].set_title('ECG Raw')
    axes[0, 0].set_ylabel('ADC'); axes[0, 0].grid(alpha=0.3)
 
    axes[0, 1].plot(t_ecg, ecg_filt, color='#922b21', lw=0.6)
    axes[0, 1].set_title('ECG Filtered (BP 0.5–40 Hz + Notch 50 Hz)')
    axes[0, 1].set_ylabel('ADC'); axes[0, 1].grid(alpha=0.3)
 
    axes[1, 0].plot(t_eda, eda_raw,  color='#2980b9', lw=0.8)
    axes[1, 0].set_title('EDA Raw (ADC)')
    axes[1, 0].set_ylabel('ADC'); axes[1, 0].set_xlabel('s'); axes[1, 0].grid(alpha=0.3)
 
    axes[1, 1].plot(t_eda, eda_filt, color='#1a5276', lw=0.8)
    axes[1, 1].set_title('EDA Filtered (normalized + LP 5 Hz)')
    axes[1, 1].set_ylabel('[0–1]'); axes[1, 1].set_xlabel('s'); axes[1, 1].grid(alpha=0.3)
 
    plt.tight_layout()
    fname = os.path.join(plot_dir, f"{subj_name}_trial{trial_idx+1:02d}.png")
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close(fig)
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    root = Tk(); root.withdraw()
    dataset_dir = filedialog.askdirectory(title="Select the AMIGOS root folder")
 
    mat_files = []
    for dirpath, _, filenames in os.walk(dataset_dir):
        for f in sorted(filenames):
            if f.endswith('.mat'):
                mat_files.append((dirpath, f))
    mat_files.sort()
 
    print(f"Found {len(mat_files)} .mat files\n")
    if not mat_files:
        print("No .mat files found.")
        return
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)
 
    print(f"neurokit2 version: {nk.__version__}\n")
 
    all_rows = []
 
    for subj_idx, (dirpath, fname) in enumerate(mat_files):
        subj_name = os.path.basename(dirpath) or f"Subject_{subj_idx+1:02d}"
        mat_path  = os.path.join(dirpath, fname)
        print(f"[{subj_idx+1}/{len(mat_files)}] {subj_name}")
 
        try:
            mat = load_mat(mat_path)
            trials, valences, arousals = get_trials_and_labels(mat)
        except Exception as e:
            print(f"  ✗ Loading error: {e}")
            continue
 
        print(f"  Trials with labels: {len(trials)}")
        plot_saved = False
 
        for trial_idx, trial_data in enumerate(trials):
            if trial_data.ndim != 2 or trial_data.shape[1] <= EDA_CH:
                continue
 
            ecg_raw = trial_data[:, ECG_CH].copy()
            eda_raw = trial_data[:, EDA_CH].copy()
 
            # Resample EDA from 256 Hz -> 128 Hz
            eda_raw_resamp = signal.resample(eda_raw, int(len(eda_raw) * EDA_FS / ECG_FS))
 
            ecg_filt = filter_ecg(ecg_raw)
            eda_filt = filter_eda(eda_raw_resamp)
 
            # Save plot only for the first valid trial per subject
            if not plot_saved:
                save_plot(ecg_raw, ecg_filt, eda_raw_resamp, eda_filt,
                          subj_name, trial_idx, plot_dir)
                plot_saved = True
 
            ecg_quality = compute_ecg_quality(ecg_filt)
            eda_quality = compute_eda_quality(eda_filt)
            feat_ecg    = ecg_features(ecg_filt)
            feat_eda    = eda_features(eda_filt)
 
            all_rows.append({
                'subject':      subj_name,
                'trial':        trial_idx + 1,
                'valence':      valences[trial_idx],
                'arousal':      arousals[trial_idx],
                'ecg_quality':  ecg_quality,
                'eda_quality':  eda_quality,
                **feat_ecg,
                **feat_eda,
            })
 
        print(f"  ✓ Done")
 
    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(OUTPUT_DIR, "amigos_features.csv")
        df.to_csv(csv_path, index=False)
 
        print(f"\n{'='*55}")
        print(f"CSV saved  : {csv_path}")
        print(f"Plots saved: {plot_dir}")
        print(f"Total rows : {len(df)}  |  Features: {df.shape[1] - 6}")
        print(f"\nPreview:")
        cols = ['subject','trial','valence','arousal',
                'ecg_quality','eda_quality',
                'ecg_hr_mean','ecg_rmssd','eda_scl_mean','eda_scr_count']
        print(df[cols].head(6).to_string(index=False))
    else:
        print("No data extracted.")
 
if __name__ == '__main__':
    main()
