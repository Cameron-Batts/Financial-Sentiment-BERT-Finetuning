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

## Original Test Results
| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Naive Bayes | 0.75 | 0.66 | 0.74 |
| Logistic regression | 0.76 | 0.64 | 0.74 |
| Off-the-shelf BERT | 0.65 | 0.51 | 0.61 |
| Frozen BERT | 0.68 | 0.61 | 0.68 |
| Fully fine-tuned BERT | 0.87 | 0.84 | 0.87 |

FinBERT reported 0.89 accuracy, 0.88 macro F1 and 0.89 weighted F1, but its training history may overlap this Financial PhraseBank evaluation corpus. Treat it as a separate, potentially contaminated comparison. Exact overlap was not audited.

Full fine-tuning improved rounded accuracy by about 11 percentage points over logistic regression. These are retained reports from the supplied run, not newly reproduced outcomes.

## Verified baselines and accelerator preparation
On October 22, 2026, the original dataset, sample and split were reproduced. Naive Bayes achieved 0.7464 accuracy / 0.6565 macro F1; logistic regression achieved 0.7608 accuracy / 0.6396 macro F1, matching their historical rounded results. See [baseline validation](reports/baseline_rerun_metrics.json). The dataset contains eight duplicate text rows, and two unique test sentences also occur in training. The original split was preserved for reproduction; these overlaps limit independent evaluation.

BERT checkpoint loading and one CPU forward/backward batch completed. Label order was confirmed as negative, neutral, positive. Full training was not rerun: a 32-example, 128-token CPU training batch took 10.53 seconds, suggesting over an hour for the full experiment. The accelerator workflow now pins model revisions, sets seed 42, selects by validation loss, retains at most two checkpoints per training run, and exports aggregate metrics. Both Trainer configurations initialized successfully on CPU with verified freeze states. GPU/MPS training and the Colab launcher remain untested here. See [preflight checks](reports/accelerator_preflight.json). Historical BERT metrics above are unchanged.

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
- `notebooks/financial_sentiment_bert.ipynb`: cleaned workflow and original numerical reports.
- `reports/financial_sentiment_bert.html`: readable analysis report.
- `reports/model_comparison.csv`: rounded original metrics and evaluation notes.
- `data/README.md`: local data setup and attribution.
- `requirements.txt`: general dependencies.
- `requirements-reproduction.txt`: pinned local reproduction packages.
- `notebooks/financial_sentiment_colab.ipynb`: GPU launcher.
- `reports/baseline_rerun_metrics.json`: recomputed traditional-classifier metrics.
- `validation/verify_baselines.py`: repeatable baseline verification (`python validation/verify_baselines.py`).
- `portfolio_card.html`: matching portfolio card.

## Validation and Limitations
Traditional baselines rerun; BERT training remains historical. Accelerator setup syntax-checked and CPU model setup tested. A single, non-stratified random split does not control duplicated/related news or temporal shift. Test labels include 51 negative, 297 neutral and 137 positive examples. Audit checkpoint label ordering: changing label metadata does not reorder pretrained output weights. A pinned local reproduction environment and fixed model revisions are now provided; Colab uses its existing PyTorch build and records versions. External-model training overlap is possible. No financial returns, production results or future-news accuracy are claimed.

## Sources
Financial PhraseBank: Malo et al. (2014), *Good debt or bad debt: Detecting semantic orientations in economic texts*, https://doi.org/10.1002/asi.23062 . Checkpoint identifiers: `MarieAngeA13/Sentiment-Analysis-BERT` and `ProsusAI/finbert`. Required source identifiers and citations are retained. Observe applicable dataset and checkpoint terms.

## Author
Cameron Batts  
GitHub: https://github.com/Cameron-Batts  
Portfolio: https://cameron-batts.github.io
