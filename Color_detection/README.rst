.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/Color_detection.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/Color_detection
    .. image:: https://readthedocs.org/projects/Color_detection/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://Color_detection.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/Color_detection/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/Color_detection
    .. image:: https://img.shields.io/pypi/v/Color_detection.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/Color_detection/
    .. image:: https://img.shields.io/conda/vn/conda-forge/Color_detection.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/Color_detection
    .. image:: https://pepy.tech/badge/Color_detection/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/Color_detection
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/Color_detection

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

===============
Color_detection
===============

Unpack and add the dataset folder to the Color_detection folder!!!!

To run this project, you need to set up a Python virtual environment and install the required dependencies. The .venv folder is not included in the repository to keep the project lightweight.
Steps to set up the environment from scratch:

Open your terminal in the project folder.
Run the following commands to create the environment and install libraries:

python -m venv .venv
.\.venv\Scripts\Activate
pip install tensorflow numpy matplotlib scikit-learn pillow pyscaffold

#########################################################################
Here are the commands to execute the main parts of the project:

To train the model:
python -m src.color_detection.train

To predict a color from a specific image:
python -m src.color_detection.predict "C:\Path\To\Your\Photo.jpg"

To run other utility modules: You can execute other scripts (like visualize, data_cleaning, or convert_tflite) by following the same syntax pattern:
python -m src.color_detection.<module_name>

####################################################

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
