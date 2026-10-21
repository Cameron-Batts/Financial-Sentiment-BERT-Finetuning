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

## Run
```bash
python -m pip install -r requirements.txt
jupyter lab notebooks/financial_sentiment_bert.ipynb
```
Place a permitted copy of `all-data.csv` in `data/`; the uploaded `all-data(1).csv` can be renamed locally. Model weights download at runtime. GPU acceleration is recommended; training and checkpoint saving require time, memory and disk. No trained checkpoints are included.

## Files
- `notebooks/financial_sentiment_bert.ipynb`: cleaned workflow and original numerical reports.
- `reports/financial_sentiment_bert.html`: readable analysis report.
- `reports/model_comparison.csv`: rounded original metrics and evaluation notes.
- `data/README.md`: local data setup and attribution.
- `requirements.txt`: dependencies.
- `portfolio_card.html`: matching portfolio card.

## Validation and Limitations
Code syntax checked; not rerun. A single, non-stratified random split does not control duplicated/related news or temporal shift. Test labels include 51 negative, 297 neutral and 137 positive examples. Audit checkpoint label ordering: changing label metadata does not reorder pretrained output weights. Dependencies/model revisions are unpinned. External-model training overlap is possible. No financial returns, production results or future-news accuracy are claimed.

## Sources
Financial PhraseBank: Malo et al. (2014), *Good debt or bad debt: Detecting semantic orientations in economic texts*, https://doi.org/10.1002/asi.23062 . Checkpoint identifiers: `MarieAngeA13/Sentiment-Analysis-BERT` and `ProsusAI/finbert`. Required source identifiers and citations are retained. Observe applicable dataset and checkpoint terms.

## Author
Cameron Batts  
GitHub: https://github.com/Cameron-Batts  
Portfolio: https://cameron-batts.github.io
