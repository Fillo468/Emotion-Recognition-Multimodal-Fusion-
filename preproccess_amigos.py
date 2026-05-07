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
import scipy.stats as stats
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
ECG_FS = 256   # Hz
EDA_FS = 128   # Hz
ECG_CH = 14
EDA_CH = 16

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "amigos_output")


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD .MAT
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
    trials_raw  = mat['joined_data']
    labels_raw  = mat['labels_selfassessment']
    trials, valences, arousals = [], [], []

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
    """
    1) Detrend polinomiale (grado 5) per rimuovere baseline wander lenta
    2) Bandpass 0.5–40 Hz (Butterworth ordine 4)
    3) Notch 50 Hz per rimozione interferenza rete elettrica
    """
    # Detrend: remove DC 
    ecg = signal.detrend(ecg, type='linear')
    t   = np.arange(len(ecg))
    poly_coeffs = np.polyfit(t, ecg, 5)
    ecg = ecg - np.polyval(poly_coeffs, t)

    nyq  = fs / 2.0
    b, a = signal.butter(4, [0.5/nyq, 40.0/nyq], btype='band')
    ecg  = signal.filtfilt(b, a, ecg)

    b2, a2 = signal.iirnotch(50.0/nyq, Q=30)
    return signal.filtfilt(b2, a2, ecg)


def filter_eda(eda, fs=EDA_FS):
    """
    1) Normalization ADC -> [0,1]
    2) Remotion spike  (window 0.5 s)
    3) Lowpass 5 Hz (Butterworth order 4)
    """
    # Normalization
    eda_min, eda_max = eda.min(), eda.max()
    eda_norm = (eda - eda_min) / (eda_max - eda_min) if eda_max > eda_min else eda.copy()

    win = int(fs * 0.5) | 1  
    from scipy.ndimage import median_filter
    eda_despike = median_filter(eda_norm, size=win)

    # Lowpass
    nyq  = fs / 2.0
    b, a = signal.butter(4, 5.0/nyq, btype='low')
    return signal.filtfilt(b, a, eda_despike)


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY WEIGHT  
# ─────────────────────────────────────────────────────────────────────────────

def _snr_estimate(sig):
    f, pxx = signal.welch(sig, nperseg=min(256, len(sig)//2))
    if len(f) < 2:
        return 0.0
    signal_power = np.sum(pxx[f < 10])
    noise_power  = np.sum(pxx[f >= 10])
    if noise_power == 0:
        return 1.0
    snr = signal_power / (signal_power + noise_power)
    return float(np.clip(snr, 0, 1))


def compute_quality(ecg_filt, eda_filt, fs_ecg=ECG_FS, fs_eda=EDA_FS):
   
    # ── ECG quality ──
    ecg_snr = _snr_estimate(ecg_filt)
    try:
        _, info   = nk.ecg_process(ecg_filt, sampling_rate=fs_ecg)
        rpeaks    = info["ECG_R_Peaks"]
        if len(rpeaks) >= 2:
            rr_ms     = np.diff(rpeaks) / fs_ecg * 1000.0
            rr_valid  = np.sum((rr_ms >= 300) & (rr_ms <= 1500)) / len(rr_ms)
        else:
            rr_valid  = 0.0
    except Exception:
        rr_valid = 0.0

    ecg_quality = float(np.clip(ecg_snr * rr_valid, 0, 1))

    # ── EDA quality ──
    eda_snr   = _snr_estimate(eda_filt)
    eda_range = float(np.clip(np.ptp(eda_filt), 0, 1))   # già in [0,1]
    eda_quality = float(np.clip(eda_snr * (0.5 + 0.5 * eda_range), 0, 1))

    # ── Combined ──
    signal_quality = 0.6 * ecg_quality + 0.4 * eda_quality
    return round(float(signal_quality), 4)


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE ENTROPY  (nonlinear EDA)
# ─────────────────────────────────────────────────────────────────────────────

def sample_entropy(sig, m=2, r_factor=0.2):
   
    N = len(sig)
    if N < 50:
        return np.nan
    r = r_factor * np.std(sig)
    if r == 0:
        return np.nan

    def _count_templates(m):
        count = 0
        for i in range(N - m):
            template = sig[i:i+m]
            for j in range(i+1, N - m):
                if np.max(np.abs(template - sig[j:j+m])) < r:
                    count += 1
        return count

    
    def _count_fast(m):
        count = 0
        for i in range(N - m):
            diffs = np.abs(sig[i+1:N-m+1] - sig[i])
            matches = np.all(
                np.column_stack([
                    np.abs(sig[i:N-m] - sig[i:N-m]) <= r
                    for _ in range(m)
                ]), axis=1
            ) if m > 1 else (diffs <= r)
            count += np.sum(matches)
        return count

    step = max(1, N // 500)
    sub  = sig[::step]
    M    = len(sub)
    B, A = 0, 0
    for i in range(M - m):
        tmpl_m  = sub[i:i+m]
        tmpl_m1 = sub[i:i+m+1]
        for j in range(i+1, M - m):
            if np.max(np.abs(tmpl_m  - sub[j:j+m]))   < r:
                B += 1
            if np.max(np.abs(tmpl_m1 - sub[j:j+m+1])) < r:
                A += 1

    if B == 0 or A == 0:
        return np.nan
    return float(-np.log(A / B))


def permutation_entropy(sig, order=3, delay=1):
 
    N = len(sig)
    if N < order * delay + 1:
        return np.nan

    from itertools import permutations
    import math

    perms = {}
    for i in range(N - (order - 1) * delay):
        pattern = tuple(np.argsort(sig[i:i + order * delay:delay]))
        perms[pattern] = perms.get(pattern, 0) + 1

    total = sum(perms.values())
    probs = [c / total for c in perms.values()]
    pe    = -sum(p * np.log(p) for p in probs if p > 0)
    max_pe = np.log(math.factorial(order))
    return float(pe / max_pe) if max_pe > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ECG
# ─────────────────────────────────────────────────────────────────────────────

def ecg_features(ecg_filt, fs=ECG_FS):
    try:
        _, info = nk.ecg_process(ecg_filt, sampling_rate=fs)
        rpeaks  = info["ECG_R_Peaks"]
        if len(rpeaks) < 4:
            raise ValueError("Troppo pochi picchi R")
        rr = np.diff(rpeaks) / fs * 1000.0   # ms

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
    except Exception:
        return {k: np.nan for k in
                ['ecg_hr_mean', 'ecg_sdnn', 'ecg_rmssd', 'ecg_pnn50',
                 'ecg_mean_rr', 'ecg_lf_power', 'ecg_hf_power', 'ecg_lf_hf']}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EDA  (time-domain + frequency-domain + nonlinear)
# ─────────────────────────────────────────────────────────────────────────────

def eda_features(eda_filt, fs=EDA_FS):
    feats = {}

    try:
        eda_signals, info = nk.eda_process(eda_filt, sampling_rate=fs)
        tonic   = eda_signals['EDA_Tonic'].values
        phasic  = eda_signals['EDA_Phasic'].values
        scr_idx = np.array(info.get('SCR_Peaks', []))
        n_scr   = len(scr_idx)
        dur_min = len(eda_filt) / fs / 60.0

        # ── Time-domain ──────────────────────────────────────────────────────
        feats['eda_mean']        = float(np.mean(eda_filt))
        feats['eda_std']         = float(np.std(eda_filt))
        feats['eda_skewness']    = float(stats.skew(eda_filt))
        feats['eda_kurtosis']    = float(stats.kurtosis(eda_filt))

        feats['eda_scl_mean']    = float(np.mean(tonic))
        feats['eda_scl_std']     = float(np.std(tonic))
        feats['eda_scl_range']   = float(np.ptp(tonic))

        feats['eda_phasic_mean'] = float(np.mean(phasic))
        feats['eda_phasic_max']  = float(np.max(phasic))
        feats['eda_phasic_auc']  = float(np.trapz(np.abs(phasic)) / fs)  # area under curve

        feats['eda_scr_count']   = int(n_scr)
        feats['eda_scr_rate']    = float(n_scr / dur_min) if dur_min > 0 else np.nan

        
        if n_scr > 0 and 'SCR_Onsets' in info and 'SCR_Amplitude' in info:
            onsets    = np.array(info['SCR_Onsets'])
            amplitudes= np.array(info['SCR_Amplitude'])
            min_len = min(len(onsets), len(scr_idx))
            if min_len > 0:
                rise_times = (scr_idx[:min_len] - onsets[:min_len]) / fs
                feats['eda_scr_rise_mean'] = float(np.mean(rise_times[rise_times > 0])) \
                                              if np.any(rise_times > 0) else np.nan
            else:
                feats['eda_scr_rise_mean'] = np.nan

            feats['eda_scr_amp_mean'] = float(np.mean(amplitudes))
            feats['eda_scr_amp_max']  = float(np.max(amplitudes))

            half_win = int(fs * 1.5)
            areas = []
            for pk in scr_idx:
                start = max(0, pk - half_win)
                end   = min(len(phasic), pk + half_win)
                areas.append(np.trapz(np.abs(phasic[start:end])) / fs)
            feats['eda_scr_area_mean'] = float(np.mean(areas))
        else:
            for k in ['eda_scr_rise_mean', 'eda_scr_amp_mean',
                      'eda_scr_amp_max', 'eda_scr_area_mean']:
                feats[k] = np.nan

        # ── Frequency-domain ─────────────────────────────────────────────────

        nperseg = min(512, len(eda_filt) // 2)
        f_pxx, pxx = signal.welch(eda_filt, fs=fs, nperseg=nperseg)

        symp_mask  = (f_pxx >= 0.045) & (f_pxx <= 0.25)
        total_mask = f_pxx > 0
        symp_power = float(np.trapz(pxx[symp_mask], f_pxx[symp_mask])) \
                     if symp_mask.any() else np.nan
        total_power= float(np.trapz(pxx[total_mask], f_pxx[total_mask])) \
                     if total_mask.any() else np.nan

        feats['eda_symp_power']  = symp_power
        feats['eda_total_power'] = total_power
        feats['eda_symp_ratio']  = float(symp_power / total_power) \
                                   if (total_power and total_power > 0) else np.nan

        if total_mask.any():
            feats['eda_peak_freq'] = float(f_pxx[np.argmax(pxx)])
        else:
            feats['eda_peak_freq'] = np.nan

        # ── Nonlinear ─────────────────────────────────────────────────────────
        feats['eda_tonic_sampen']  = sample_entropy(tonic)
        feats['eda_phasic_permen'] = permutation_entropy(phasic)

    except Exception as e:
        nan_keys = [
            'eda_mean', 'eda_std', 'eda_skewness', 'eda_kurtosis',
            'eda_scl_mean', 'eda_scl_std', 'eda_scl_range',
            'eda_phasic_mean', 'eda_phasic_max', 'eda_phasic_auc',
            'eda_scr_count', 'eda_scr_rate',
            'eda_scr_rise_mean', 'eda_scr_amp_mean', 'eda_scr_amp_max', 'eda_scr_area_mean',
            'eda_symp_power', 'eda_total_power', 'eda_symp_ratio', 'eda_peak_freq',
            'eda_tonic_sampen', 'eda_phasic_permen',
        ]
        feats = {k: np.nan for k in nan_keys}

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# PLOT raw vs filtered
# ─────────────────────────────────────────────────────────────────────────────

def save_plot(ecg_raw, ecg_filt, eda_raw, eda_filt, subj_name, trial_idx, plot_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle(f"{subj_name} – Trial {trial_idx + 1}", fontsize=13, fontweight='bold')

    t_ecg = np.arange(len(ecg_raw)) / ECG_FS
    t_eda = np.arange(len(eda_raw)) / EDA_FS

    axes[0, 0].plot(t_ecg, ecg_raw,  color='#e74c3c', lw=0.6)
    axes[0, 0].set_title('ECG Raw'); axes[0, 0].set_ylabel('ADC'); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t_ecg, ecg_filt, color='#922b21', lw=0.6)
    axes[0, 1].set_title('ECG Filtrato (detrend + BP 0.5–40 Hz + Notch 50 Hz)')
    axes[0, 1].set_ylabel('ADC'); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t_eda, eda_raw,  color='#2980b9', lw=0.8)
    axes[1, 0].set_title('EDA Raw (ADC)'); axes[1, 0].set_ylabel('ADC')
    axes[1, 0].set_xlabel('s'); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(t_eda, eda_filt, color='#1a5276', lw=0.8)
    axes[1, 1].set_title('EDA Filtrata (normalizzata + despike + LP 5 Hz)')
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
    dataset_dir = filedialog.askdirectory(title="Seleziona la cartella principale AMIGOS")
    if not dataset_dir:
        print("Nessuna cartella selezionata. Uscita.")
        return

    mat_files = []
    for dirpath, _, filenames in os.walk(dataset_dir):
        for f in sorted(filenames):
            if f.endswith('.mat'):
                mat_files.append((dirpath, f))
    mat_files.sort()

    print(f"Trovati {len(mat_files)} file .mat\n")
    if not mat_files:
        print("Nessun file .mat trovato."); return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    all_rows = []

    for subj_idx, (dirpath, fname) in enumerate(mat_files):
        subj_name = os.path.basename(dirpath) or f"Subject_{subj_idx+1:02d}"
        mat_path  = os.path.join(dirpath, fname)
        print(f"[{subj_idx+1}/{len(mat_files)}] {subj_name}")

        try:
            mat = load_mat(mat_path)
            trials, valences, arousals = get_trials_and_labels(mat)
        except Exception as e:
            print(f"  ✗ Errore caricamento: {e}"); continue

        print(f"  Trial con etichette: {len(trials)}")
        plot_saved = False

        for trial_idx, trial_data in enumerate(trials):
            if trial_data.ndim != 2 or trial_data.shape[1] <= EDA_CH:
                continue

            ecg_raw = trial_data[:, ECG_CH].copy()
            eda_raw = trial_data[:, EDA_CH].copy()

            # Ricampiona EDA da ECG_FS (256 Hz) a EDA_FS (128 Hz)
            eda_raw_resamp = signal.resample(eda_raw, int(len(eda_raw) * EDA_FS / ECG_FS))

            ecg_filt = filter_ecg(ecg_raw)
            eda_filt = filter_eda(eda_raw_resamp)

            # Plot solo per il primo trial di ogni soggetto
            if not plot_saved:
                save_plot(ecg_raw, ecg_filt, eda_raw_resamp, eda_filt,
                          subj_name, trial_idx, plot_dir)
                plot_saved = True

            # Quality weight (0-1)
            quality = compute_quality(ecg_filt, eda_filt)

            feat_ecg = ecg_features(ecg_filt)
            feat_eda = eda_features(eda_filt)

            all_rows.append({
                'subject':        subj_name,
                'trial':          trial_idx + 1,
                'valence':        valences[trial_idx],
                'arousal':        arousals[trial_idx],
                'signal_quality': quality,   # <- peso per late fusion
                **feat_ecg,
                **feat_eda,
            })

        print(f"  ✓ OK")

    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(OUTPUT_DIR, "amigos_features.csv")
        df.to_csv(csv_path, index=False)

        print(f"\n{'='*60}")
        print(f"CSV salvato : {csv_path}")
        print(f"Plot salvati: {plot_dir}")
        print(f"Righe totali: {len(df)}  |  Feature totali: {df.shape[1] - 5}")
        print(f"\nAnteprima (colonne chiave):")
        cols = ['subject', 'trial', 'valence', 'arousal', 'signal_quality',
                'ecg_hr_mean', 'ecg_rmssd',
                'eda_scl_mean', 'eda_scr_count', 'eda_symp_power',
                'eda_tonic_sampen', 'eda_phasic_permen']
        available = [c for c in cols if c in df.columns]
        print(df[available].head(6).to_string(index=False))
    else:
        print("Nessun dato estratto.")


if __name__ == '__main__':
    main()
