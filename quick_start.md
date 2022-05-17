# Quick Start Guide

## Overview

The plugin is designed for obtaining accurate vessel segmentation from 3D lightsheet fluorescent images. Overall, the plugin is organized as follows:

vessel-express-napari
    |
    |--> Parameter Tuning
    |--> Evaluation 

**Parameter Tuning**: allows a user to configure a segmentation workflow either from scratch or by testing preset configurations. **Evaluation**: allows a user to easily loop through all images to inspect (in 3D) whether the segmentation is good or failed, or the raw image has major issues.


## Parameter Tuning

You could start with running a preset configuration for a specific organ. Please note that it is very possible a preset configuration may not work for your own data, even from the same type of organ. Make sure you always check the results and adjust each step when necessary (move your mouse to each function and each parameter will show tool tips). 

First of all, let's have a very brief overview of main concepts in the segmentation workflows. An input image will go through three main stages: Pre-processing, Core Segmentation, Post-processing. **Pre-processing** includes resizing (making the image isotropic *coming soon*) and smoothing (reducing noise). **Core Segmentation** contains two main functions a core thresholding (capturing the vessels of very high intensity) and a vesselness filter (capturing vessels of different thickness and different contrasts). The results from the core thresholding and the vesselness filter(s) will be merged. **Post-processing** contains a list of operations to refine the segmentations (e.g., a closing step to remove small gaps near vessel intersections, a topology-preserving thinning step to reduce over-segmentation, etc.) Users can select whcih to use according to their data, or even skip all of them.


When you want to test on a new image, here are the steps we would recommend:

1. Select a preset configuration for a specfic organ (if not existing, choose "muscle", which is a very basic workflow to start with), then click "Run Preset".
2. After about 20~40 seconds (the large the image is, the longer it will take), a list of layers will show up, where the layer names represent the parameters used in each step. Take "muscle" for example. Five new layers will be created: "smoothed_image" (result of smoothing), "threshold_3.5" (result of the core threshold with scale = 3.5), "ves_1_70_threshold_li" (result of vesselness filter with sigma=1, gamma=70, cutoff_method=threshold_li), "merged_segmentation" (the result of merging the core threshold result and the vesselness filter result), "cleaned_100" (the result of applying post-cleaning with min_size=100).
3. Now, you can adjust the paramters if necessary. For example, if you see those "bulky" very thick and bright vessels are not fully segmented, you could reduce the *scale* in the core threshold step from 3.5 to 3 or 2.5 to capture more. If you see the segmentation does not do well on vessels relatively thick, you could add another vesselness filter with larger sigma value to improve the performance on thicker vessels. By making differey layers visible/invisible, you will be able to know how the segmentation looks by combining which layers. If you find a good combination, for example threshold_3, ves_1_70_threshold_li, ves_2_10_threshold_otsu, then make sure to re-run the merge step to generate a merged segmentation (so that you can apply post-processing on it). 
4. After adjusting the parameters, make sure to close the layers do not belong to your final workflow (e.g., you tested the core thresholding step with scale=2.5 and scale=3, and you find scale=3 is good, make sure close the threshold_2 layer). This is meant to inform the plugin which set of functions and parameters you finally choose to use. Then click "Generate Config file" (coming soon ... not done yet).


## Evaluation

The evaluation part is designed to inspect the output from the vessel express snakemake pipeline, where in the same folder, you can find both the raw images with name XXXXX.tif (or XXXXX.tiff) and the segmentations Binary_XXXXX.tiff. After you select this folder in the selection box, the first image and its segmentation will be automatically displayed (raw image in gray with adjusted contrast and the segmentation in magenta). Then, you can choose "Good", "Failed", or "Bad Image", then click "Next". You can continue this process until the last one. At the end, click "Save", you can save the inspection results in a CSV file.


## More tips

### changing from front view to side view. 

On the lower left corner of the napari window, you may two special buttons "Change the order of visible axes" (a box with an arrow pointing to the right) and "Transpose order of the last two visible axes" (a box with an arrow pointing to the left). The first one will change from front view to side view (YZ or XZ), while the second one allows to switch Y and X in front view. Note: the axes of an image has order ZYX in napari.

### check skeleton visualization

You can peek what the skeleton may look like by running "skeletonization" on the final segmentation layer. Note: this is only a sneak-peak. The final skeleton will go through further pruning to refine the extracted structure.