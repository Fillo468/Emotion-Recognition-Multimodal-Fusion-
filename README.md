# Emotion-Recognition-Multimodal-Fusion-

## Preprocessing — EMOVOME 

The EMOVOME dataset provides a large set of pre-extracted vocal/acoustic features from 999 unique, spontaneous voice messages from spanish speakers from Whatsapp, together with multiple annotations for valence and arousal provided by the speakers themselves and different listeners.

Our preprocessing pipeline for EMOVOME dataset is structured as follows:

1. Feature-label merging

The acoustic feature table is first merged with the corresponding annotation labels in order to obtain a unified sample-level representation.
We fill missing value with the median of the feature

2. Aggregation of listener annotations

Each sample in EMOVOME contains multiple listener annotations.

For both valence and arousal, the final target value is computed as the mean score across annotators, producing a single continuous target per emotional dimension.

3. Feature selection using mutual information

EMOVOME contains a high-dimensional acoustic feature space.

To reduce redundancy and retain the most informative descriptors, mutual information is computed independently for:

* valence
* arousal

For each target, the 15 most informative features are selected.

4. Clean dataset generation

The selected features are used to create a reduced and cleaner dataset.

This step helps mitigate the curse of dimensionality, reduces irrelevant variability, and provides a more compact representation for consequent machine learning experiments.

---

## Preprocessing — AMIGOS 

From the AMIGOS dataset, only the ECG and EDA (GSR) physiological signals are used.

For each trial, self-assessment annotations are extracted and used as emotional targets:

* Arousal
* Valence

The preprocessing pipeline is structured as follows:

1. Trial extraction and label association

Each .mat file is analysed to extract:

trial-level physiological recordings (joined_data)
corresponding self-assessment labels (labels_selfassessment)

Only trials containing valid annotations are retained.

For every valid trial, the following target values are associated:

* arousal
* valence

2. Signal selection

From the original multichannel recording, only the following channels are retained:

* ECG → channel 14
* EDA / GSR → channel 16

The original AMIGOS recordings are sampled at different effective rates in our pipeline:

- ECG: 256 Hz
- EDA: resampled from 256 Hz to 128 Hz

3. ECG preprocessing

The ECG signal undergoes two filtering stages:

- band-pass filter: 0.5 – 40 Hz
- notch filter: 50 Hz

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

The final output is a structured feature table used for machine learning experiments.

This produces a compact physiological representation of the AMIGOS dataset, ready to be compared and fused with the vocal features extracted from the EMOVOME dataset.

---

## Regression — EMOVOME (Vocal Features)

After preprocessing and feature selection, the regression experiments on the EMOVOME dataset are performed using the selected vocal acoustic features.


1. Input dataset

The regression stage uses the cleaned EMOVOME dataset obtained after preprocessing.

The dataset contains:

- 999 samples
- 24 selected vocal features

Non-predictive columns such as sample identifiers and target labels are removed before model training.

2. Target normalization

To make EMOVOME compatible with the physiological targets extracted from the AMIGOS dataset dataset, emotional annotations are rescaled.

The original EMOVOME target values:

{-2, -1, 0, 1, 2}

are mapped to the AMIGOS-compatible range:

{1, 3, 5, 7, 9}

using the transformation:

y = 2x + 5

This alignment allows direct comparison between vocal and physiological regression outputs.

3. Train / validation / test split

The dataset is divided into three disjoint subsets:

* Training set: 60%
* Validation set: 20%
* Test set: 20%

A fixed random seed (random_state = 42) is used to ensure reproducibility.

4. Feature standardization

All vocal features are standardized using z-score normalization.

The scaler is fitted only on the training set and then applied to validation and test sets.

This ensures consistent feature scaling across the entire regression pipeline.

5. Regression models

Two independent Random Forest regressors are trained:

- Arousal regressor
n_estimators = 500
max_depth = None
min_samples_split = 5
bootstrap = True

- Valence regressor
n_estimators = 100
max_depth = 10
min_samples_split = 10
bootstrap = True

These hyperparameters were selected through preliminary *grid search* optimization.

Separate models are trained for the two emotional dimensions because arousal and valence may exhibit different predictive patterns.

6. Model persistence

The trained vocal regressors are saved as serialized models for later reuse in the late fusion stage:

modello_voce_arousal.pkl
modello_voce_valence.pkl

7. Evaluation metrics

Performance is evaluated on both validation and test sets using the following regression metrics:

* MSE — Mean Squared Error
* RMSE — Root Mean Squared Error
* MAE — Mean Absolute Error
* R² — Coefficient of Determination

8. Test performance


The results indicate that vocal features provide a stronger predictive signal for arousal than for valence in the EMOVOME dataset.

---
