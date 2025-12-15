# tests/test_converter.py
from color_detection.convert_tflite import representative_data_gen

def test_representative_data_gen(tf):
    gen = representative_data_gen()
    batch = next(gen)

    assert isinstance(batch, list)
    assert batch[0].shape[0] == 1
    assert batch[0].dtype == tf.float32
