# Die Funktion _smoothing aufrufen

import numpy as np
from napari import Viewer
from vessel_express import ParameterTuning
from tifffile import imread

viewer = Viewer(show = False)
para_tuning = ParameterTuning(viewer)

file1 = 'vessel_express\\_tests\\images\\Raw_liver_1.tiff'
image1 = imread(file1)
image2 = para_tuning._smoothing(preset = True, data = image1)
np.save('smoothing.npy', image2)
print('done')
