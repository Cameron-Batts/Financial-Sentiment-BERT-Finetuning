"""Recompute the original split and traditional classifiers without model downloads."""
import contextlib
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import numpy
import pandas as pd
import sklearn
from sklearn.metrics import classification_report

root = Path(__file__).resolve().parents[1]
os.chdir(root)
notebook = json.loads((root/'notebooks/financial_sentiment_bert.ipynb').read_text())
scope = {}
for cell in notebook['cells']:
    if cell.get('id') in {f'financial-bert-{i:03}' for i in [1,3,5,6,7,8,9]}:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(''.join(cell['source']), cell['id'], 'exec'), scope)
data = pd.read_csv(root/'data/all-data.csv', names=['sentiment','content'], encoding='latin-1')
assert len(data) == 4846 and not data.isna().any().any()
assert set(data.sentiment) == {'negative','neutral','positive'}
assert [len(scope[n]) for n in ['X_train','X_val','X_test']] == [1453,485,485]
assert scope['y_test'].value_counts().to_dict() == {'neutral':297,'positive':137,'negative':51}
reports = {name: classification_report(scope['y_test'], scope[key], output_dict=True, zero_division=0)
           for name,key in [('Naive Bayes','y_pred_nb'),('Logistic regression','y_pred_lr')]}
assert round(reports['Naive Bayes']['accuracy'],2) == .75
assert round(reports['Logistic regression']['accuracy'],2) == .76
record = {
    'date':'2026-10-22','python':platform.python_version(),'pandas':pd.__version__,
    'numpy':numpy.__version__,'sklearn':sklearn.__version__,
    'dataset_sha256':hashlib.sha256((root/'data/all-data.csv').read_bytes()).hexdigest(),
    'dataset_rows':len(data),'sample_rows':len(scope['news_df']),
    'train_rows':len(scope['X_train']),'validation_rows':len(scope['X_val']),'test_rows':len(scope['X_test']),
    'test_class_counts':scope['y_test'].value_counts().to_dict(),
    'duplicates_in_full_data':int(data.content.duplicated().sum()),
    'unique_test_texts_also_in_train':len(set(scope['X_test']) & set(scope['X_train'])),
    'reports':reports,
    'scope':'Traditional classifiers only. No transformer training, inference or model downloads.'
}
(root/'reports/baseline_rerun_metrics.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({name:{'accuracy':v['accuracy'],'macro_f1':v['macro avg']['f1-score']} for name,v in reports.items()},indent=2))
print('Unique test texts also in training:',record['unique_test_texts_also_in_train'])
