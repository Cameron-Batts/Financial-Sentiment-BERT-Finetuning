"""Check aggregate consistency of the supplied MPS export, not raw predictions."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root/'reports/gpu_rerun_metrics.json'
data = json.loads(path.read_text())
baseline = json.loads((root/'reports/baseline_rerun_metrics.json').read_text())
assert data['dataset_sha256'] == baseline['dataset_sha256']
assert data['test_rows'] == 485
assert data['bert_revision'] == '938fb4267e118e6f892d9825253fd28893d95966'
assert data['finbert_revision'] == '0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031'
for name, report in data['models'].items():
    labels = ['negative','neutral','positive']
    assert [report[x]['support'] for x in labels] == [51,297,137]
    correct = sum(report[x]['recall']*report[x]['support'] for x in labels)
    assert math.isclose(correct, round(correct))
    assert math.isclose(correct/485, report['accuracy'])
    for metric in ['precision','recall','f1-score']:
        assert math.isclose(sum(report[x][metric] for x in labels)/3, report['macro avg'][metric])
        assert math.isclose(sum(report[x][metric]*report[x]['support'] for x in labels)/485, report['weighted avg'][metric])
for name in ['Naive Bayes','Logistic regression']:
    assert data['models'][name] == baseline['reports'][name]
selected = {}
for name in ['frozen','full']:
    logs = data['training_logs'][name]
    evaluations = [x for x in logs if 'eval_loss' in x]
    assert [x['epoch'] for x in evaluations] == list(range(1,11))
    assert logs[-1]['epoch'] == 10 and logs[-1]['step'] == 460
    best = min(evaluations, key=lambda x:x['eval_loss'])
    assert data['best_validation_checkpoints'][name].endswith('checkpoint-'+str(best['step']))
    selected[name] = {'epoch':best['epoch'],'step':best['step'],'validation_loss':best['eval_loss']}
record = {
    'checked_at_utc':datetime.now(timezone.utc).isoformat(),
    'export_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'source':'User-supplied MPS rerun export; this session did not execute the training runs.',
    'checks':'Dataset hash/revisions, exact traditional baselines, class supports, aggregate arithmetic, ten evaluation epochs, and minimum-validation-loss checkpoint selection agree.',
    'selected_checkpoints':selected,
    'limitation':'Raw predictions, model weights and the executed notebook were not supplied; aggregate consistency does not independently verify execution or per-example outputs.'
}
(root/'reports/rerun_validation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
