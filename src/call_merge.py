# Die Funktion _merge aufrufen

import numpy as np
from napari import Viewer
from vessel_express import ParameterTuning

viewer = Viewer(show = False)
para_tuning = ParameterTuning(viewer)

file1 = 'vessel_express\\_tests\\images\\threshold.npy'
file2 = 'vessel_express\\_tests\\images\\ves_li.npy'
file3 = 'vessel_express\\_tests\\images\\ves_otsu.npy'
file4 = 'vessel_express\\_tests\\images\\ves_triangle.npy'

image1 = np.load(file1)
image2 = np.load(file2)
image3 = np.load(file3)
image4 = np.load(file4)

image5 = para_tuning._merge(preset = True, layers = 2, data1 = image1,
    data2 = image2)
image6 = para_tuning._merge(preset = True, layers = 3, data1 = image1,
    data2 = image3, data3 = image4)

np.save('merge_2layers.npy', image5)
np.save('merge_3layers.npy', image6)
print('done')
