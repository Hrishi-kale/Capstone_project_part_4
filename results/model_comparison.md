| Model | Accuracy | Precision | Recall | F1 Score | Notes |
|---|---|---|---|---|---|
| Random Forest (supervised) | 0.9419 | 0.9467 | 0.9404 | 0.9435 | Trained on labelled data; precision/recall computed for the phishing(-1) class |
| Isolation Forest (unsupervised) | 0.4410 | 0.4576 | 0.4470 | 0.4523 | Never saw labels during training; contamination parameter set to 0.5000 (train-set phishing proportion) |
