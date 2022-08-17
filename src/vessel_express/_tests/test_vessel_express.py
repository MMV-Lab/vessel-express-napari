import pytest
import numpy as np
from tifffile import imread
from napari.layers import Image
from vessel_express import ParameterTuning


@pytest.mark.smoothing
def test_smoothing(make_napari_viewer):
    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)     # 1. Objekt dieser Klasse

    file1 = 'src/vessel_express/_tests/images/Raw_liver_1.tiff'
    file3 = 'src/vessel_express/_tests/images/smoothing.npy'
    image1 = imread(file1)
    image3 = np.load(file3)

    image2 = para_tuning._smoothing(preset = True, data = image1)
    assert np.array_equal(image2, image3)


@pytest.mark.add_image
def test_add_image(make_napari_viewer):
    # 05.07.2022
    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)

    file1 = 'src/vessel_express/_tests/images/Raw_liver_1.tiff'
    image1 = imread(file1)

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
    # 06.07.2022
    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)

    file1 = 'src/vessel_express/_tests/images/Raw_liver_1.tiff'
    file3 = 'src/vessel_express/_tests/images/isotropic_linux.npy'
    image1 = imread(file1)
    image3 = np.load(file3)
    
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
    # 08.07.2022
    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)

    file1 = 'src/vessel_express/_tests/images/Raw_liver_1.tiff'
    file3 = 'src/vessel_express/_tests/images/threshold.npy'
    image1 = imread(file1)
    image3 = np.load(file3)
    
    image2 = para_tuning._threshold(preset = True, image = image1, scale = 2.0)
    assert np.array_equal(image2, image3)


@pytest.mark.vesselness
def test_vesselness(make_napari_viewer):
    # 12.07.2022
    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)

    file1 = 'src/vessel_express/_tests/images/Raw_liver_1.tiff'
    file3 = 'src/vessel_express/_tests/images/ves_li.npy'
    file4 = 'src/vessel_express/_tests/images/ves_otsu.npy'
    file5 = 'src/vessel_express/_tests/images/ves_triangle.npy'
    image1 = imread(file1)
    image3 = np.load(file3)
    image4 = np.load(file4)
    image5 = np.load(file5)

    image2 = para_tuning._vesselness(preset = True, image = image1, sigma = 2,
        gamma = 10, cutoff_method = 'threshold_li')
    assert np.array_equal(image2, image3)

    image2 = para_tuning._vesselness(preset = True, image = image1, sigma = 2,
        gamma = 10, cutoff_method = 'threshold_otsu')
    assert np.array_equal(image2, image4)

    image2 = para_tuning._vesselness(preset = True, image = image1, sigma = 2,
        gamma = 10, cutoff_method = 'threshold_triangle')
    assert np.array_equal(image2, image5)


@pytest.mark.merge
def test_merge(make_napari_viewer):
    # 14.07.2022
    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)

    file1 = 'src/vessel_express/_tests/images/threshold.npy'
    file2 = 'src/vessel_express/_tests/images/ves_li.npy'
    file3 = 'src/vessel_express/_tests/images/ves_otsu.npy'
    file4 = 'src/vessel_express/_tests/images/ves_triangle.npy'
    file5 = 'src/vessel_express/_tests/images/merge_2layers.npy'
    file6 = 'src/vessel_express/_tests/images/merge_3layers.npy'

    image1 = np.load(file1)
    image2 = np.load(file2)
    image3 = np.load(file3)
    image4 = np.load(file4)
    image5 = np.load(file5)
    image6 = np.load(file6)

    image7 = para_tuning._merge(preset = True, layers = 2, data1 = image1,
        data2 = image2)
    image8 = para_tuning._merge(preset = True, layers = 3, data1 = image1,
        data2 = image3, data3 = image4)

    assert np.array_equal(image7, image5)
    assert np.array_equal(image8, image6)


@pytest.mark.closing
def test_closing(make_napari_viewer):
    # 14.07.2022
    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)

    file1 = 'src/vessel_express/_tests/images/merge_2layers.npy'
    file3 = 'src/vessel_express/_tests/images/closing.npy'
    image1 = np.load(file1)
    image3 = np.load(file3)

    image2 = para_tuning._closing(preset = True, image = image1, kernel = 5)
    assert np.array_equal(image2, image3)


@pytest.mark.hole_removal
def test_hole_removal(make_napari_viewer):
    # 15.07.2022
    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)

    file1 = 'src/vessel_express/_tests/images/closing.npy'
    file3 = 'src/vessel_express/_tests/images/hole_removal.npy'
    image1 = np.load(file1)
    image3 = np.load(file3)

    image2 = para_tuning._hole_removal(preset = True, image = image1,
        max_size = 10)
    assert np.array_equal(image2, image3)


@pytest.mark.thinning
def test_thinning(make_napari_viewer):
    # 15.07.2022
    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)

    file1 = 'src/vessel_express/_tests/images/hole_removal.npy'
    file3 = 'src/vessel_express/_tests/images/thinning.npy'
    image1 = np.load(file1)
    image3 = np.load(file3)

    image2 = para_tuning._thinning(preset = True, image = image1,
        min_thickness = 1, thin = 1)
    assert np.array_equal(image2, image3)


@pytest.mark.cleaning
def test_cleaning(make_napari_viewer):
    # 15.07.2022
    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)

    file1 = 'src/vessel_express/_tests/images/thinning.npy'
    file3 = 'src/vessel_express/_tests/images/cleaning.npy'
    image1 = np.load(file1)
    image3 = np.load(file3)

    image2 = para_tuning._cleaning(preset = True, image = image1,
        min_size = 100)
    assert np.array_equal(image2, image3)


@pytest.mark.skeleton
def test_skeleton(make_napari_viewer):
    # 15.07.2022
    viewer = make_napari_viewer()
    para_tuning = ParameterTuning(viewer)

    file1 = 'src/vessel_express/_tests/images/cleaning.npy'
    file3 = 'src/vessel_express/_tests/images/skeleton.npy'
    image1 = np.load(file1)
    image3 = np.load(file3)

    image2 = para_tuning._skeleton(preset = True, image = image1)
    assert np.array_equal(image2, image3)
