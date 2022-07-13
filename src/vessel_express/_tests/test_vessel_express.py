import pytest
import numpy as np
from tifffile import imread
from napari.layers import Image
from vessel_express import ParameterTuning

@pytest.mark.smoothing
def test_smoothing(make_napari_viewer):
    file1 = 'vessel_express\\_tests\\images\\Raw_liver_1.tiff'
    file3 = 'vessel_express\\_tests\\images\\smoothing.npy'
    image1 = imread(file1)
    image3 = np.load(file3)

    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)     # 1. Objekt dieser Klasse

    image2 = para_tuning._smoothing(preset = True, data = image1)
    assert np.array_equal(image2, image3)

@pytest.mark.add_image
def test_add_image(make_napari_viewer):
    file1 = 'vessel_express\\_tests\\images\\Raw_liver_1.tiff'
    image1 = imread(file1)

    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)

    viewer.add_image(data=image1, name='Raw_liver_1')
    string1 = para_tuning.c_isotropic.currentText()

    for layer in viewer.layers:
        if layer.name == 'Raw_liver_1' and type(layer) == Image:
            image2 = layer.data
            break
    
    assert string1 == 'Raw_liver_1'
    assert np.array_equal(image1, image2)


@pytest.mark.isotropic
def test_isotropic(make_napari_viewer):
    file1 = 'vessel_express\\_tests\\images\\Raw_liver_1.tiff'
    file3 = 'vessel_express\\_tests\\images\\isotropic.npy'
    image1 = imread(file1)
    image3 = np.load(file3)
    
    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)
    viewer.add_image(data=image1, name='Raw_liver_1')
    
    # li_x, li_y und li_z sind Objekte vom Typ QLineEdit
    para_tuning.li_x.setText('0.108')
    para_tuning.li_y.setText('0.108')
    para_tuning.li_z.setText('0.29')
    
    # Die Methode _isotropic() benötigt ein 3D-Image sowie die Daten x, y und z
    para_tuning._isotropic()
    
    for layer in viewer.layers:
        if layer.name == 'isotropic_0.108_0.108_0.29' and type(layer) == Image:
            image2 = layer.data
            break
    
    assert np.array_equal(image2, image3)

@pytest.mark.threshold
def test_threshold(make_napari_viewer):
    file1 = 'vessel_express\\_tests\\images\\Raw_liver_1.tiff'
    file3 = 'vessel_express\\_tests\\images\\threshold.npy'
    image1 = imread(file1)
    image3 = np.load(file3)
    
    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)

    image2 = para_tuning._threshold(preset = True, image = image1, scale = 2.0)
    assert np.array_equal(image2, image3)

@pytest.mark.vesselness
def test_vesselness(make_napari_viewer):
    file1 = 'vessel_express\\_tests\\images\\Raw_liver_1.tiff'
    file3 = 'vessel_express\\_tests\\images\\ves_li.npy'
    file4 = 'vessel_express\\_tests\\images\\ves_otsu.npy'
    file5 = 'vessel_express\\_tests\\images\\ves_triangle.npy'
    image1 = imread(file1)
    image3 = np.load(file3)
    image4 = np.load(file4)
    image5 = np.load(file5)

    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)

    image2 = para_tuning._vesselness(preset = True, image = image1, sigma = 2,
        gamma = 10, cutoff_method = 'threshold_li')
    assert np.array_equal(image2, image3)

    image2 = para_tuning._vesselness(preset = True, image = image1, sigma = 2,
        gamma = 10, cutoff_method = 'threshold_otsu')
    assert np.array_equal(image2, image4)

    image2 = para_tuning._vesselness(preset = True, image = image1, sigma = 2,
        gamma = 10, cutoff_method = 'threshold_triangle')
    assert np.array_equal(image2, image5)
