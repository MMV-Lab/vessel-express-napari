# Die Funktion _thinning aufrufen

import numpy as np
from napari import Viewer
from vessel_express import ParameterTuning

viewer = Viewer(show = False)
para_tuning = ParameterTuning(viewer)

file1 = 'vessel_express\\_tests\\images\\hole_removal.npy'
image1 = np.load(file1)
image2 = para_tuning._thinning(preset = True, image = image1, min_thickness = 1,
    thin = 1)

np.save('thinning.npy', image2)
print('done')
