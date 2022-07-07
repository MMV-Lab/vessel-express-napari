import pytest
import numpy as np
from tifffile import imread
from napari.layers import Image
from vessel_express import ParameterTuning

@pytest.mark.smoothing
def test_smoothing(make_napari_viewer):
    file1 = 'vessel_express\\_tests\\images\\Raw_liver_1.tiff'
    file3 = 'vessel_express\\_tests\\images\\smoothed_Image.tif'
    image1 = imread(file1)
    image3 = imread(file3)

    viewer = make_napari_viewer()
    p_tuning = ParameterTuning(viewer)     # 1. Objekt dieser Klasse

    image2 = p_tuning._smoothing(preset = True, data = image1)
    assert np.array_equal(image2, image3)

@pytest.mark.add_image
def test_add_image(make_napari_viewer):
    file1 = 'vessel_express\\_tests\\images\\Raw_liver_1.tiff'
    image1 = imread(file1)

    viewer = make_napari_viewer()
    p_tuning = ParameterTuning(viewer)

    viewer.add_image(data=image1, name='Raw_liver_1')
    string1 = p_tuning.c_isotropic.currentText()

    for layer in viewer.layers:
        if layer.name == 'Raw_liver_1' and type(layer) == Image:
            image2 = layer.data
            break
    
    assert string1 == 'Raw_liver_1'
    assert np.array_equal(image1, image2)


@pytest.mark.isotropic
def test_isotropic(make_napari_viewer):
    file1 = 'vessel_express\\_tests\\images\\Raw_liver_1.tiff'
    file3 = 'vessel_express\\_tests\\images\\isotropic_0.108_0.108_0.29.tif'
    image1 = imread(file1)
    image3 = imread(file3)
    
    viewer = make_napari_viewer()
    p_tuning = ParameterTuning(viewer)
    viewer.add_image(data=image1, name='Raw_liver_1')
    
    # li_x, li_y und li_z sind Objekte vom Typ QLineEdit
    p_tuning.li_x.setText('0.108')
    p_tuning.li_y.setText('0.108')
    p_tuning.li_z.setText('0.29')
    
    # Die Methode _isotropic() benötigt ein 3D-Image sowie die Daten x, y und z
    p_tuning._isotropic()
    
    for layer in viewer.layers:
        if layer.name == 'isotropic_0.108_0.108_0.29' and type(layer) == Image:
            image2 = layer.data
            break
    
    assert np.array_equal(image2, image3)
