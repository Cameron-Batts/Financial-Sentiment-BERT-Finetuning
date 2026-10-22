# Financial Sentiment Classification with BERT Fine-Tuning

**Author:** Cameron Batts  
**Tools:** Python · PyTorch · Hugging Face Transformers · scikit-learn · pandas

## Business Problem
Classify financial-news sentiment while evaluating minority-class performance and the benefit of adapting a general sentiment model to domain-specific text.

## Project Approach
- Sample 2,423 of 4,846 financial-news sentences.
- Split into 1,453 training, 485 validation and 485 test examples.
- Train unigram/bigram Naive Bayes and logistic regression benchmarks.
- Compare off-the-shelf BERT, classification-head-only training and full fine-tuning.
- Use ten epochs, batch size 32, epoch-level validation and best-validation-loss checkpoint selection.
- Inspect external FinBERT separately because training-corpus overlap may affect its evaluation.

## MPS rerun results — October 22, 2026
| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Naive Bayes | 0.7464 | 0.6565 | 0.7385 |
| Logistic regression | 0.7608 | 0.6396 | 0.7365 |
| Off-the-shelf BERT | 0.6536 | 0.5105 | 0.6080 |
| Frozen BERT | 0.6866 | 0.6101 | 0.6774 |
| Fine-tuned BERT | 0.8412 | 0.8199 | 0.8439 |
| External FinBERT (possible training overlap) | 0.8907 | 0.8778 | 0.8924 |

The user-supplied MPS export records ten completed epochs for head-only adaptation and full fine-tuning on the original 485-example test split. Fine-tuned BERT reached 0.8412 accuracy, 0.8199 macro F1 and 0.8439 weighted F1, compared with historical rounded values 0.87, 0.84 and 0.87. The rerun improves accuracy by 8.04 percentage points over logistic regression. Validation loss selected epoch 1 for the full model (0.3956); later validation losses increased as training loss fell, indicating overfitting. Head-only adaptation selected epoch 10. Aggregate arithmetic, dataset hash, checkpoint revisions, baseline reports and logged checkpoint selection were checked. Raw predictions, model weights and the executed notebook were not supplied, so the training was not independently replayed here. Two unique test sentences also occur in training, and external FinBERT may have seen the evaluation corpus.

The traditional classifiers also matched an independent local rerun exactly. See [supplied MPS metrics](reports/gpu_rerun_metrics.json), [aggregate checks](reports/rerun_validation.json), and [baseline validation](reports/baseline_rerun_metrics.json). The full-model training logged 312.6 seconds and the frozen model 126.2 seconds on MPS. The Colab launcher remains untested here.

## Original Test Results
| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Naive Bayes | 0.75 | 0.66 | 0.74 |
| Logistic regression | 0.76 | 0.64 | 0.74 |
| Off-the-shelf BERT | 0.65 | 0.51 | 0.61 |
| Frozen BERT | 0.68 | 0.61 | 0.68 |
| Fully fine-tuned BERT | 0.87 | 0.84 | 0.87 |

FinBERT reported 0.89 accuracy, 0.88 macro F1 and 0.89 weighted F1, but its training history may overlap this Financial PhraseBank evaluation corpus. Treat it as a separate, potentially contaminated comparison. Exact overlap was not audited.

These rounded reports describe the original run; the new MPS export above is kept separately. The original run environment and causes of the differences were not isolated experimentally.


## Run on your Mac
```bash
python -m pip install -r requirements-reproduction.txt
python -m pip install jupyterlab
jupyter lab notebooks/financial_sentiment_bert.ipynb
```
Place a permitted copy of `all-data.csv` in `data/`; the uploaded `all-data(1).csv` can be renamed locally. Model weights download at runtime. GPU acceleration is recommended; training and checkpoint saving require time, memory and disk. No trained checkpoints are included.

## Run in a GPU notebook
Open [the Colab launcher](https://colab.research.google.com/github/Cameron-Batts/Financial-Sentiment-BERT-Finetuning/blob/main/notebooks/financial_sentiment_colab.ipynb), select a GPU runtime, then run the setup cell and upload the original CSV when prompted. Continue through all project cells. Restart the runtime if package changes require it, then rerun setup. The launcher preserves Colab's GPU PyTorch build and records actual versions; that environment may differ from the pinned local setup. GPU availability and runtime duration depend on your Colab account.

On a Mac with MPS, use the main notebook. It detects MPS, CUDA or CPU, and stops before BERT inference/training if no accelerator is available. After all experiments finish, the final cell writes `reports/gpu_rerun_metrics.json`; download that file before a hosted runtime expires. Neither data nor model weights are uploaded to GitHub by the notebook.

## Files
- `notebooks/financial_sentiment_bert.ipynb`: workflow and reports formatted from the supplied MPS export.
- `reports/financial_sentiment_bert.html`: readable analysis report.
- `reports/model_comparison.csv`: rounded original metrics and evaluation notes.
- `reports/gpu_rerun_metrics.json`: original user-supplied MPS export.
- `reports/rerun_model_comparison.csv`: precise rerun metrics.
- `reports/rerun_validation.json`: aggregate validation and evidence limits.
- `validation/verify_rerun.py`: repeatable aggregate validation (`python validation/verify_rerun.py`).
- `data/README.md`: local data setup and attribution.
- `requirements.txt`: general dependencies.
- `requirements-reproduction.txt`: pinned local reproduction packages.
- `notebooks/financial_sentiment_colab.ipynb`: GPU launcher.
- `reports/baseline_rerun_metrics.json`: recomputed traditional-classifier metrics.
- `validation/verify_baselines.py`: repeatable baseline verification (`python validation/verify_baselines.py`).
- `portfolio_card.html`: matching portfolio card.

## Validation and Limitations
Traditional baselines independently rerun; the user-supplied MPS export records completed BERT training and was checked for aggregate consistency. The original run remains preserved for comparison. A single, non-stratified random split does not control duplicated/related news or temporal shift. Test labels include 51 negative, 297 neutral and 137 positive examples. Checkpoint label order was checked against the explicit mapping; changing metadata alone does not reorder pretrained output weights. A pinned local reproduction environment and fixed model revisions are now provided; Colab uses its existing PyTorch build and records versions. External-model training overlap is possible. No financial returns, production results or future-news accuracy are claimed.

## Sources
Financial PhraseBank: Malo et al. (2014), *Good debt or bad debt: Detecting semantic orientations in economic texts*, https://doi.org/10.1002/asi.23062 . Checkpoint identifiers: `MarieAngeA13/Sentiment-Analysis-BERT` and `ProsusAI/finbert`. Required source identifiers and citations are retained. Observe applicable dataset and checkpoint terms.

## Author
Cameron Batts  
GitHub: https://github.com/Cameron-Batts  
Portfolio: https://cameron-batts.github.io
