# Die Funktion _cleaning aufrufen

import numpy as np
from napari import Viewer
from vessel_express import ParameterTuning

viewer = Viewer(show = False)
para_tuning = ParameterTuning(viewer)

file1 = 'vessel_express\\_tests\\images\\thinning.npy'
image1 = np.load(file1)
image2 = para_tuning._cleaning(preset = True, image = image1, min_size = 100)

np.save('cleaning.npy', image2)
print('done')
