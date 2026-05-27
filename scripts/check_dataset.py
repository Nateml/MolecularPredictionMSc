from ..config import config
from src.data.datasets import dataset_from_schema

data_path = config.data.resolved_dataset_path()
schema_path = config.data.resolved_schema_path()
print('data_path=', data_path)
print('schema_path=', schema_path)
dataset = dataset_from_schema(data_path, schema_path)
print('dataset type:', type(dataset).__name__)
print('len(dataset)=', len(dataset))
print('index len=', len(getattr(dataset, '_index', [])))
print('first index=', getattr(dataset, '_index', [])[:1])
print('has records?', hasattr(dataset,'records'))
try:
    r = dataset._load_record(0)
    print('first record keys:', list(r.keys()))
except Exception as e:
    print('error loading first record:', e)
