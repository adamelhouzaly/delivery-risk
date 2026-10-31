"""Select on validation, report final test once, persist the entire pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import FEATURES, NUMERIC, CATEGORICAL, validate, temporal_split

def metrics(y, probabilities, threshold=.5):
    predictions = probabilities >= threshold
    return {'average_precision': float(average_precision_score(y, probabilities)),
            'precision': float(precision_score(y, predictions, zero_division=0)),
            'recall': float(recall_score(y, predictions, zero_division=0)),
            'confusion_matrix_tn_fp_fn_tp': confusion_matrix(y, predictions, labels=[0, 1]).ravel().tolist()}

def train(csv_path, out_dir, provenance):
    raw = Path(csv_path).read_bytes()
    df = validate(pd.read_csv(csv_path))
    training, validation, test = temporal_split(df)
    candidates = {
        'dummy': DummyClassifier(strategy='prior'),
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'forest': RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=10, random_state=42, n_jobs=1),
    }
    fitted, scores = {}, {}
    for name, estimator in candidates.items():
        preprocessing = ColumnTransformer([
            ('numeric', StandardScaler(), NUMERIC),
            ('category', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        ])
        pipeline = Pipeline([('preprocess', preprocessing), ('model', estimator)])
        pipeline.fit(training[FEATURES], training.late)
        scores[name] = metrics(validation.late, pipeline.predict_proba(validation[FEATURES])[:, 1])
        fitted[name] = pipeline
    selected = max(scores, key=lambda name: scores[name]['average_precision'])
    pipeline = fitted[selected]
    probabilities = pipeline.predict_proba(test[FEATURES])[:, 1]
    version = hashlib.sha256(raw).hexdigest()[:12] + '-' + selected
    report = {
        'data_provenance': provenance, 'data_sha256': hashlib.sha256(raw).hexdigest(),
        'model_version': version, 'selected_model': selected, 'threshold': .5,
        'selection_rule': 'Highest validation average precision; threshold fixed at 0.5',
        'split': {name: {'rows': len(part), 'start': str(part.ordered_at.min()),
                         'end': str(part.ordered_at.max()), 'late_rate': float(part.late.mean())}
                  for name, part in zip(['train', 'validation', 'test'], [training, validation, test])},
        'validation': scores, 'test': metrics(test.late, probabilities),
        'test_by_region': {},
        'limitations': ['No real-world claim follows from synthetic data.',
                       'No causal interpretation; no calibrated probability guarantee.',
                       'Only training data fits preprocessing and estimators.',
                       'No refit after model selection; test partition is not used for tuning.'],
    }
    for region in sorted(test.region.unique()):
        mask = (test.region == region).to_numpy()
        report['test_by_region'][str(region)] = {
            'rows': int(mask.sum()), 'late_rate': float(test.late[mask].mean()),
            'recall': float(recall_score(test.late[mask], probabilities[mask] >= .5, zero_division=0))}
    errors = test[['order_id', 'region', 'late']].copy()
    errors['risk'] = probabilities
    errors['predicted_late'] = probabilities >= .5
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': pipeline, 'version': version, 'threshold': .5,
                 'data_provenance': provenance}, out / 'model.joblib')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    errors[errors.late != errors.predicted_late].to_csv(out / 'errors.csv', index=False)
    return report

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', default='artifacts')
    p.add_argument('--provenance', required=True, help='e.g. synthetic-demo or dataset URL and license')
    args = p.parse_args()
    print(json.dumps(train(args.csv, args.out, args.provenance), indent=2))

if __name__ == '__main__':
    main()
