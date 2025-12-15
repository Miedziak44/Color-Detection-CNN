# tests/test_prediction.py
from color_detection.predict import get_class_names

def test_get_class_names():
    classes = get_class_names()

    assert isinstance(classes, list)
    assert len(classes) > 0
