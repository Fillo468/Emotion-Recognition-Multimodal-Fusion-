# Emotion-Recognition-Multimodal-Fusion-

## Preprocessing — EMOVOME 

The EMOVOME dataset provides a large set of pre-extracted vocal/acoustic features for each voice message, together with multiple annotations for valence and arousal provided by different listeners.

Our preprocessing pipeline for EMOVOME datset is structured as follows:

1. Feature-label merging

The acoustic feature table is first merged with the corresponding annotation labels in order to obtain a unified sample-level representation.
We fill missing value with the median of the feature

2. Target scale alignment

Since the project compares EMOVOME with AMIGOS dataset, the emotional annotations are rescaled to make the two datasets comparable.

The original EMOVOME labels are mapped to the same numerical range used in AMIGOS:

* -2 → 1
* -1 → 3
* 0 → 5
* 1 → 7
* 2 → 9

This normalization preserves the ordinal structure of the labels while aligning the emotional scales across datasets.

3. Aggregation of listener annotations

Each sample in EMOVOME contains multiple listener annotations.

For both valence and arousal, the final target value is computed as the mean score across annotators, producing a single continuous target per emotional dimension.

4. Feature selection using mutual information

EMOVOME contains a high-dimensional acoustic feature space.

To reduce redundancy and retain the most informative descriptors, mutual information is computed independently for:

* valence
* arousal

For each target, the 15 most informative features are selected.

5. Clean dataset generation

The selected features are used to create a reduced and cleaner dataset.

This step helps mitigate the curse of dimensionality, reduces irrelevant variability, and provides a more compact representation for downstream machine learning experiments.

---

## Preprocessing — AMIGOS 

From the AMIGOS dataset, only the ECG and EDA (GSR) physiological signals are used.

For each trial, self-assessment annotations are extracted and used as emotional targets:

* Arousal
* Valence

The preprocessing pipeline is structured as follows.

1. Trial extraction and label association

Each .mat file is parsed to extract:

trial-level physiological recordings (joined_data)
corresponding self-assessment labels (labels_selfassessment)

Only trials containing valid annotations are retained.

For every valid trial, the following target values are associated:

arousal
valence
2. Signal selection

From the original multichannel recording, only the following channels are retained:

ECG → channel 14
EDA / GSR → channel 16

The original AMIGOS recordings are sampled at different effective rates in our pipeline:

ECG: 256 Hz
EDA: resampled from 256 Hz to 128 Hz
3. ECG preprocessing

The ECG signal undergoes two filtering stages:

band-pass filter: 0.5 – 40 Hz
notch filter: 50 Hz

This step removes low-frequency drift, high-frequency noise, and power-line interference.

4. EDA preprocessing

EDA values are first min-max normalized into the range [0,1].

After normalization, a low-pass filter at 5 Hz is applied to remove high-frequency noise while preserving the slower electrodermal dynamics.

5. Signal quality estimation

A signal quality score is computed for each trial in order to estimate the reliability of the extracted physiological segment.

The quality score combines ECG and EDA information:

- ECG quality
* spectral SNR estimate
* validity of RR intervals extracted from detected R-peaks
- EDA quality
* spectral SNR estimate
* effective dynamic range of the filtered signal


The resulting value ranges from 0 (poor quality) to 1 (high quality).

6. ECG feature extraction

From the filtered ECG signal, heart rate variability (HRV) features are extracted.

The final ECG feature set includes:

* mean heart rate
* SDNN
* RMSSD
* pNN50
* mean RR interval
* low-frequency power (LF)
* high-frequency power (HF)
* LF/HF ratio

These descriptors capture both time-domain and frequency-domain cardiac dynamics.

7. EDA feature extraction

The filtered EDA signal is decomposed into:

- tonic component
- phasic component

From these components, the following features are extracted:

* mean EDA
* standard deviation
* mean tonic level (SCL)
* tonic standard deviation
* tonic range
* mean phasic activity
* maximum phasic activity
* number of skin conductance responses (SCR)
* SCR rate

These features summarize both baseline autonomic activation and event-related electrodermal responses.

8. Final dataset generation

For each valid trial, the extracted physiological features are combined with:

* subject ID
* trial index
* valence
* arousal
* signal quality score

The final output is a structured feature table used for downstream machine learning experiments.

This produces a compact physiological representation of the AMIGOS dataset, ready to be compared and fused with the vocal features extracted from the EMOVOME dataset.

---
