import pandas as pd
import pytest
from fastapi.testclient import TestClient
from delivery_risk.data import generate, validate, temporal_split, FEATURES
from delivery_risk.train import train
from delivery_risk.api import create_app

@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp('model')
    path = root / 'demo.csv'
    generate(1200).to_csv(path, index=False)
    return root, train(path, root, 'synthetic-demo')

def test_no_leakage_or_overlapping_timestamps():
    df = generate(1000)
    df.loc[1, 'ordered_at'] = df.loc[0, 'ordered_at']
    a, b, c = temporal_split(validate(df))
    assert a.ordered_at.max() < b.ordered_at.min() < c.ordered_at.min()
    assert b.ordered_at.max() < c.ordered_at.min()
    assert not {'late', 'ordered_at', 'order_id'} & set(FEATURES)

@pytest.mark.parametrize('mutation', ['duplicate', 'null', 'negative', 'label'])
def test_bad_training_data(mutation):
    df = generate(100)
    if mutation == 'duplicate': df.loc[1, 'order_id'] = df.loc[0, 'order_id']
    if mutation == 'null': df.loc[0, 'carrier'] = None
    if mutation == 'negative': df.loc[0, 'distance_km'] = -1
    if mutation == 'label': df.loc[0, 'late'] = 2
    with pytest.raises(ValueError): validate(df)

def test_training_and_inference(trained):
    root, report = trained
    assert sum(v['rows'] for v in report['split'].values()) == 1200
    assert report['selected_model'] in report['validation']
    assert 0 <= report['test']['average_precision'] <= 1
    client = TestClient(create_app(root / 'model.joblib'))
    order = {'distance_km': 400, 'items': 2, 'promised_hours': 48,
             'backlog': 80, 'carrier': 'previously-unseen', 'region': 'north'}
    response = client.post('/predict', json={'orders': [order]})
    assert response.status_code == 200
    assert 0 <= response.json()['predictions'][0]['late_risk'] <= 1
    assert client.get('/ready').json()['data_provenance'] == 'synthetic-demo'
    assert client.post('/predict', json={'orders': [{**order, 'late': 1}]}).status_code == 422
    assert client.post('/predict', json={'orders': [{**order, 'distance_km': -1}]}).status_code == 422
    assert client.post('/predict', json={'orders': []}).status_code == 422
    assert client.get('/metrics').json()['errors'] >= 3

def test_missing_model(tmp_path):
    client = TestClient(create_app(tmp_path / 'absent.joblib'))
    assert client.get('/health').status_code == 200
    assert client.get('/ready').status_code == 503
