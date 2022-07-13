# Die Funktion _vesselness aufrufen

import numpy as np
from napari import Viewer
from vessel_express import ParameterTuning
from tifffile import imread

viewer = Viewer(show = False)
para_tuning = ParameterTuning(viewer)

file1 = 'vessel_express\\_tests\\images\\Raw_liver_1.tiff'
image1 = imread(file1)

image2 = para_tuning._vesselness(preset = True, image = image1, sigma = 2,
    gamma = 10, cutoff_method = 'threshold_li')
image3 = para_tuning._vesselness(preset = True, image = image1, sigma = 2,
    gamma = 10, cutoff_method = 'threshold_otsu')
image4 = para_tuning._vesselness(preset = True, image = image1, sigma = 2,
    gamma = 10, cutoff_method = 'threshold_triangle')

np.save('ves_li.npy', image2)
np.save('ves_otsu.npy', image3)
np.save('ves_triangle.npy', image4)
print('done')
