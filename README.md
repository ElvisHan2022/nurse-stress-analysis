# Nurse Stress Detection — Extended Analysis

Extending the analysis from [`MLMAMidtermProject`](https://github.com/adishri8/MLMAMidtermProject)
(Kwon, Guan, Shrinivasan, Momtaz — JHU EN.520.439): predicting nurse stress from wearable
sensor data (heart rate, EDA, skin temperature, movement), comparing a single global model
against per-nurse personalized models.

**Outline** Edit 1:
1. Problem: Nurse stress is reported retrospectively and is biased given time that has passed since their shift and the qualitative nature of the survey.
2. Data: The *current dataset* is a merged dataset of E4 Empatica reported biometrics (HR, EDA, movement intensity, temperature) from Dryad for a cohort of 15 nurses (non-pregnant, smoking, chronically ill). Our next dataset is a cohort of nurses from WESAD.
3. Model: We use both global (all nurses together) and per-nurse models. Each model is paired with its own curated model (ie. branched ensembles between physiological features and time, sequence-based, etc).
4. Metrics: Our metrics are F1-score, AUC, and PR-AUC. We are understanding gaps in our pre-processing techniques (merging removed features and points given that biometrics were recorded at different frequencies) and will select metrics accordingly.

**Start here: [`START_HERE.md`](START_HERE.md)** — a narrative walkthrough of the project,
written for anyone unfamiliar with the codebase or the terminology.

Other docs:
- [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) — setup instructions and a file-by-file inventory of
  the reference repo
- [`CONCEPTS.md`](CONCEPTS.md) — deep-dive on the methodology (windowing, normalization,
  metrics, model architectures)

The reference repo (read-only, not vendored here — clone it separately) lives in `reference/`.
Original work lives in `my_work/`.
