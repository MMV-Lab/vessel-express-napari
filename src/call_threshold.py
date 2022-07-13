# Die Funktion _threshold aufrufen

import numpy as np
from napari import Viewer
from vessel_express import ParameterTuning
from tifffile import imread

viewer = Viewer(show = False)
para_tuning = ParameterTuning(viewer)

file1 = 'vessel_express\\_tests\\images\\Raw_liver_1.tiff'
image1 = imread(file1)
image2 = para_tuning._threshold(preset = True, image = image1, scale = 2.0)
np.save('threshold.npy', image2)
print('done')
