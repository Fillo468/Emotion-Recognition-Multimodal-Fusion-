# Emotion-Recognition-Multimodal-Fusion-
Preprocessing — EMOVOME

The EMOVOME dataset provides a large set of pre-extracted vocal/acoustic features for each voice message, together with multiple annotations for valence and arousal provided by different listeners.

Our preprocessing pipeline for EMOVOME datset is structured as follows:

1. Feature-label merging

The acoustic feature table is first merged with the corresponding annotation labels in order to obtain a unified sample-level representation.
We fill missing value with the median of the feature

2. Target scale alignment

Since the project compares EMOVOME with AMIGOS dataset, the emotional annotations are rescaled to make the two datasets comparable.

The original EMOVOME labels are mapped to the same numerical range used in AMIGOS:

-2 → 1
-1 → 3
0 → 5
1 → 7
2 → 9

This normalization preserves the ordinal structure of the labels while aligning the emotional scales across datasets.

3. Aggregation of listener annotations

Each sample in EMOVOME contains multiple listener annotations.

For both valence and arousal, the final target value is computed as the mean score across annotators, producing a single continuous target per emotional dimension.

4. Feature selection using mutual information

EMOVOME contains a high-dimensional acoustic feature space.

To reduce redundancy and retain the most informative descriptors, mutual information is computed independently for:

valence
arousal

For each target, the 15 most informative features are selected.

5. Clean dataset generation

The selected features are used to create a reduced and cleaner dataset.

This step helps mitigate the curse of dimensionality, reduces irrelevant variability, and provides a more compact representation for downstream machine learning experiments.
