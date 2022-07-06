import pytest
import numpy as np
from tifffile import imread

from vessel_express import ParameterTuning

@pytest.mark.smoothing
def test_smoothing(make_napari_viewer):
    file1 = '\\vessel-express\\images\\Raw_liver_1.tiff'
    file3 = '\\vessel-express\\images\\smoothed_Image.tif'
    image1 = imread(file1)
    image3 = imread(file3)

    viewer = make_napari_viewer()
    p_tuning = ParameterTuning(viewer)     # 1. Objekt dieser Klasse

    image2 = p_tuning._smoothing(preset = True, data = image1)
    assert np.array_equal(image2, image3)

@pytest.mark.add_image
def test_add_image(make_napari_viewer):
    file1 = '\\vessel-express\\images\\Raw_liver_1.tiff'
    image1 = imread(file1)

    viewer = make_napari_viewer()
    p_tuning = ParameterTuning(viewer)

    viewer.add_image(data=image1, name='Raw_liver_1')
    string1 = p_tuning.c_isotropic.currentText()
    assert string1 == 'Raw_liver_1'

'''
@pytest.mark.isotropic
def test_isotropic(make_napari_viewer):
    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)     # 1. Objekt dieser Klasse

    assert True
'''