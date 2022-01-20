import itk
import numpy as np
from typing import Union
from importlib import import_module


def vesselness_filter(
    im: np.ndarray,
    dim: int,
    sigma: Union[int, float],
    cutoff_method: str
) -> np.ndarray:
    """
    function for running ITK 3D/2D vesselness filter

    Parameters:
    ------
    im: np.ndarray
        the 3D image to be applied on
    dim: int
        either apply 3D vesselness filter or apply 2D vesselness slice by slice
    sigma: Union[float, int]
        the kernal size of the filter
    cutoff_method: str
        which method to use for determining the cutoff value, options include any
        threshold method in skimage, such as "threshold_li", "threshold_otsu", 
        "threshold_triangle", etc.. See https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html

    Returns:
    ---------
    vess: np.ndarray
        filter output
    """
    if dim == 3:
        im_itk = itk.image_view_from_array(im)
        hessian_itk = itk.hessian_recursive_gaussian_image_filter(im_itk, sigma=sigma, normalize_across_scale=True)
        vess_tubulness = itk.hessian_to_objectness_measure_image_filter(hessian_itk, object_dimension=1)
        vess = np.asarray(vess_tubulness)
    elif dim ==2:
        vess = np.zeros_like(im)
        for z in range(im.shape[0]):
            im_itk = itk.image_view_from_array(im[z,:,:])
            hessian_itk = itk.hessian_recursive_gaussian_image_filter(im_itk, sigma=sigma, normalize_across_scale=True)
            vess_tubulness = itk.hessian_to_objectness_measure_image_filter(hessian_itk, object_dimension=1)
            vess_2d = np.asarray(vess_tubulness)
            vess[z, :, :] = vess_2d[:, :]

    module_name = import_module("skimage.filters")
    threshold_function = getattr(module_name, cutoff_method)

    return vess > threshold_function(vess)
