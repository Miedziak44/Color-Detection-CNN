# tests/test_dataset.py
from color_detection import dataset

def test_get_datasets_structure():
    train_ds, val_ds = dataset.get_datasets()

    # Smoke test: both datasets must be iterable
    assert hasattr(train_ds, '__iter__')
    assert hasattr(val_ds, '__iter__')
