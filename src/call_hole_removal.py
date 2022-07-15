# Die Funktion _hole_removal aufrufen

import numpy as np
from napari import Viewer
from vessel_express import ParameterTuning

viewer = Viewer(show = False)
para_tuning = ParameterTuning(viewer)

file1 = 'vessel_express\\_tests\\images\\closing.npy'
image1 = np.load(file1)
image2 = para_tuning._hole_removal(preset = True, image = image1, max_size = 10)

np.save('hole_removal.npy', image2)
print('done')
