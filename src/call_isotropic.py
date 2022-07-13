# Die Funktion _isotropic aufrufen

import numpy as np
from napari import Viewer
from napari.layers import Image
from vessel_express import ParameterTuning
from tifffile import imread

viewer = Viewer(show = False)
para_tuning = ParameterTuning(viewer)

file1 = 'vessel_express\\_tests\\images\\Raw_liver_1.tiff'
image1 = imread(file1)
viewer.add_image(data=image1, name='Raw_liver_1')

para_tuning.li_x.setText('0.108')
para_tuning.li_y.setText('0.108')
para_tuning.li_z.setText('0.29')

para_tuning._isotropic()

for layer in viewer.layers:
    if layer.name == 'isotropic_0.108_0.108_0.29' and type(layer) == Image:
        image2 = layer.data
        break

np.save('isotropic.npy', image2)
print('done')
