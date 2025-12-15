# tests/test_model.py
from color_detection import model as model_module, config

def test_model_build(tf):
    num_classes = 5
    model = model_module.build_model(num_classes)

    # Output layer units
    assert model.output_shape[-1] == num_classes

    # Forward pass
    dummy_batch = tf.random.uniform(
        (1, config.IMG_HEIGHT, config.IMG_WIDTH, 3)
    )
    out = model(dummy_batch)
    assert out.shape == (1, num_classes)
