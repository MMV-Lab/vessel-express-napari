# Die Funktion _closing aufrufen

import numpy as np
from napari import Viewer
from vessel_express import ParameterTuning

viewer = Viewer(show = False)
para_tuning = ParameterTuning(viewer)

file1 = 'vessel_express\\_tests\\images\\merge_2layers.npy'
image1 = np.load(file1)
image2 = para_tuning._closing(preset = True, image = image1, kernel = 5)

np.save('closing.npy', image2)
print('done')
