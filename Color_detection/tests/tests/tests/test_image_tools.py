# tests/test_image_tools.py
from color_detection.data_cleaning import convert_png_to_jpg, find_corrupted_images
from PIL import Image

def test_convert_png_to_jpg(tmp_path):
    test_png = tmp_path / "test.png"
    Image.new('RGB', (50, 50)).save(test_png)

    convert_png_to_jpg(directory=str(tmp_path), delete_original=True)

    assert not test_png.exists()
    assert (tmp_path / "test.jpg").exists()


def test_find_corrupted_images(tmp_path):
    corrupt = tmp_path / "bad.jpg"
    corrupt.write_bytes(b"notanimage")

    bad_files = find_corrupted_images(directory=str(tmp_path))
    assert len(bad_files) == 1
    assert str(corrupt) in bad_files
