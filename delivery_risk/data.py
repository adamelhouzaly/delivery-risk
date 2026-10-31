"""Order-time inputs only; the demo generator is explicitly synthetic."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

NUMERIC = ['distance_km', 'items', 'promised_hours', 'backlog']
CATEGORICAL = ['carrier', 'region']
FEATURES = NUMERIC + CATEGORICAL

def generate(n=4000, seed=42):
    if n < 100:
        raise ValueError('Generate at least 100 orders')
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'order_id': [f'demo-{i:06d}' for i in range(n)],
        'ordered_at': pd.date_range('2025-01-01', periods=n, freq='h'),
        'distance_km': rng.uniform(5, 1500, n).round(2),
        'items': rng.integers(1, 12, n),
        'promised_hours': rng.choice([24, 48, 72, 120], n),
        'backlog': rng.integers(0, 150, n),
        'carrier': rng.choice(['alpha', 'beta', 'gamma'], n),
        'region': rng.choice(['north', 'south', 'west'], n),
    })
    logit = (-3.0 + df.distance_km / 650 + df.backlog / 65
             - df.promised_hours / 65 + (df.carrier == 'beta') * .8)
    df['late'] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df

def validate(df):
    required = set(FEATURES + ['order_id', 'ordered_at', 'late'])
    if missing := required - set(df):
        raise ValueError(f'Missing columns: {sorted(missing)}')
    df = df[list(FEATURES + ['order_id', 'ordered_at', 'late'])].copy()
    if df.isna().any().any():
        raise ValueError('Missing values are not accepted')
    if df.order_id.duplicated().any():
        raise ValueError('Duplicate order IDs would contaminate evaluation')
    df['ordered_at'] = pd.to_datetime(df.ordered_at, utc=True, errors='raise')
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='raise')
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f'{col} must be finite and nonnegative')
    if (df['items'] < 1).any() or (df.promised_hours <= 0).any():
        raise ValueError('items and promised_hours must be positive')
    if not df.late.isin([0, 1]).all():
        raise ValueError('late must be binary')
    return df.sort_values('ordered_at').reset_index(drop=True)

def temporal_split(df):
    times = df.ordered_at.drop_duplicates().sort_values().tolist()
    if len(times) < 10:
        raise ValueError('Need at least 10 distinct timestamps')
    a, b = times[int(len(times) * .6)], times[int(len(times) * .8)]
    parts = (df[df.ordered_at < a], df[(df.ordered_at >= a) & (df.ordered_at < b)],
             df[df.ordered_at >= b])
    if any(len(p) < 20 or p.late.nunique() < 2 for p in parts):
        raise ValueError('Each temporal partition needs 20+ rows and both labels')
    return parts

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/demo.csv')
    p.add_argument('--rows', type=int, default=4000)
    args = p.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate(args.rows).to_csv(path, index=False)
    print(f'Wrote {args.rows} SYNTHETIC demo orders to {path}')

if __name__ == '__main__':
    main()
