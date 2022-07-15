# Die Funktion _skeleton aufrufen

import numpy as np
from napari import Viewer
from vessel_express import ParameterTuning

viewer = Viewer(show = False)
para_tuning = ParameterTuning(viewer)

file1 = 'vessel_express\\_tests\\images\\cleaning.npy'
image1 = np.load(file1)
image2 = para_tuning._skeleton(preset = True, image = image1)

np.save('skeleton.npy', image2)
print('done')
