from inspect import CORO_CLOSED
from PyQt5.QtWidgets import QComboBox, QLabel, QSizePolicy
import napari
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QPushButton, QSlider, QHBoxLayout, QVBoxLayout, QScrollArea, QFileDialog, QMessageBox
from qtpy.QtCore import Qt
from napari.layers import Image
from tifffile import imread

# packages required by processing functions
from .utils import vesselness_filter
import os
import numpy as np
from glob import glob
from aicssegmentation.core.pre_processing_utils import  edge_preserving_smoothing_3d
from skimage.morphology import remove_small_objects, binary_closing, cube
from aicssegmentation.core.utils import topology_preserving_thinning


class ParameterTuning(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Labels
        self.l_preset_layer = QLabel("Select Layer")
        self.l_preset_name = QLabel("Select Configuration")
        self.l_title = QLabel("<font color='green'>VesselExpress Segmentation Parameter Tuning:</font>")
        self.l_1 = QLabel("smoothing:")
        self.l_2 = QLabel("core-threshold:")
        self.l_3 = QLabel("core-vesselness:")
        self.l_4 = QLabel("merge segmentation:")
        self.l_5 = QLabel("post-closing:")
        self.l_6 = QLabel("post-thinning:")
        self.l_7 = QLabel("post-cleaning:")
        self.l_8 = QLabel("post-hole-removing:")
        self.l_9 = QLabel("skeletonization:")
        self.l_scale = QLabel("scale")
        self.l_sigma = QLabel("- sigma")
        self.l_gamma = QLabel("- gamma")
        self.l_operation_dim = QLabel("- operation_dim")
        self.l_cutoff_method = QLabel("- cutoff-method")
        self.l_threshold_result = QLabel("threshold result")
        self.l_vesselness_1 = QLabel("vesselness 1")
        self.l_vesselness_2 = QLabel("vesselness 2")
        self.l_kernel_size = QLabel("kernel-size")
        self.l_min_thick = QLabel("min_thickness")
        self.l_thin = QLabel("thin")
        self.l_min_size = QLabel("min_size")
        self.l_max_hole_size = QLabel("max hole size")

        # Set tooltips
        smoothing_layer_tip = (
            "Goal of this step: Smooth your image without losing sharp edges.<br><br>\n\n"
            "You may not need this step if (1) your image has little noise "
            "and (2) the segmentation results are accurate enough.<br><br>\n \n"
            "Parameters: \n"
            "\t<p style='margin-left: 40px'>No parameters are needed.</p>\n \n"
            "Instruction: \n"
            "\t<p style='margin-left: 40px'>Select the image to smooth and click \"Run\".</p>"
        )
        self.l_1.setToolTip(smoothing_layer_tip)
        core_thresh_tip = (
            "Goal of this step: Extract the vessels with very high intensity. Don't worry about the vessels "
            "with mid/low intensities, which will be detected by the next step <br><br>\n\n"
            "The threshold is set as intensity_mean + 'scale' x intensity_standard_deviation. <br><br>\n \n"
            "Parameters: \n"
            "\t<p style='margin-left: 40px'>scale: A large value will result in a higher threshold. </p>\n \n"
            "Instruction: \n"
            "\t<p style='margin-left: 40px'>Select the image (usually the smoothed image) to apply on, select the scale and click \"Run\".</p>"
        )
        self.l_2.setToolTip(core_thresh_tip)
        core_vessel_tip = (
            "Goal of this step: Detect filamentous objects of different thickness with vesselness filter. You can use this filter 1 or 2 times. "
            "Then you can combine the results of two different vesselness filters in the merge step.  For example, you can use one filter to "
            "catch thinner vessels and use another filter to pick up thicker vessels.<br><br>\n\n"
 
            "A Frangi filter will be applied first and then a cutoff value will be calculated"
            "to binarize the result.<br><br>\n\n"

            "Parameters: \n"
            "\t<p style='margin-left: 40px'>sigma: The kernel size of the Frangi filter. Use a large sigma for thicker vessels.<br>\n"
            "\tgamma: the gamma value in Frangi filter, larger value results in less sensitive filter.<br>\n"
            "\tcutoff-method: The method used to determine how to binarize the filter result as segmentation.</p>\n\n"
            "Instruction: \n"
            "\t<p style='margin-left: 40px'>Select the image (usually the smoothed image) to apply on, select parameters and click \"Run\".</p>"
        )
        self.l_3.setToolTip(core_vessel_tip)
        core_merge_tip = (
            "Goal of this step: Merge different segmentation results (thresholding results and up to two vesselness filter results)  "
            "to obtain the final segmenation.<br><br>\n \n"
            "In most cases none of the above indiviudal segmentation step will work perfectly as a single filter.<br>\n"
            "Each step can be responsible for segmenting different parts of the vessels and combining them may yield good results for your application.<br><br>\n \n"
            "Instruction: \n"
            "\t<p style='margin-left: 40px'>Select which version of the thresholding based method to use and select<br>\n"
            "\tup to two vesselness segmentation restuls, then click \"Run\" to see how the<br>\n"
            "\tresult looks after merging them.</p>"
        )
        self.l_4.setToolTip(core_merge_tip)
        post_close_tip = (
            "Goal of this step: The vesselness filter may have broken segmentation near junction areas. You can use this closing step to bridge such gaps.<br><br>\n \n"
            "Parameters: \n"
            "\t<p style='margin-left: 40px'>kernel_size: A large value will close larger gaps, but may falsely merge proximal vessels.</p>\n\n"
            "Instruction: \n"
            "\t<p style='margin-left: 40px'>Select the segmentation result (usually after merging) to apply on,<br>\n"
            "\tselect the kernel_size and click \"Run\".</p>"
        )
        self.l_5.setToolTip(post_close_tip)
        post_thin_tip = (
            "Goal of this step: Thin the segmentation results.<br><br>\n\n"
            "The segmentation result may look thicker than it should be due to the diffraction of light. When necessary, this thinning step can make them thinner without breaking the connectivity.<br><br>\n\n"
            "Parameters: \n"
            "<p style='margin-left: 40px'>\tmin_thickness: Any vessel thinner than this value will <b>not</b> be further thinned.<br>\n"
            "\tthin: How many pixels to thin your segmentations by.</p>"
        )
        self.l_6.setToolTip(post_thin_tip)
        self.l_7.setToolTip("Any segmented objects smaller than min_size will be removed to clean up your result.")
        self.l_8.setToolTip("remove small holes in the segmentation to avoid loops in skeleton")
        self.l_9.setToolTip("show skeleton")
        
        core_thresh_scale_tip = (
            "Larger value will result in higher threshold value,\n"
            "and therefore less pixels will be segmented."
        )
        self.l_scale.setToolTip(core_thresh_scale_tip)
        core_vessel_sigma_tip = (
            "Larger value will help you pick up thicker vessels."
        )
        self.l_sigma.setToolTip(core_vessel_sigma_tip)
        self.l_gamma.setToolTip("gamma value of Frangi filter")
        core_vessel_cutoff_tip = (
            "Choose between different thresholding method to binarize\n"
            "the filter output into segmentation result."
        )
        self.l_cutoff_method.setToolTip(core_vessel_cutoff_tip)
        self.l_threshold_result.setToolTip("Layer of the thresholding based segmentation.")
        self.l_vesselness_1.setToolTip("Layer of the primary vesselness segmentation.")
        self.l_vesselness_2.setToolTip("Layer of the secondary vesselness segmentation.")
        self.l_kernel_size.setToolTip("Large value will close larger gaps, but may falsely merge proximal vessels.")
        self.l_min_thick.setToolTip("Any vessel thinner than this value will <b>not</b> be further thinned.")
        self.l_thin.setToolTip("How many pixels to thin your vessels by.")
        self.l_min_size.setToolTip("The minimum size of segmented objects to keep.")
        self.l_max_hole_size.setToolTip("the maximum size of holes to be filled")

        # Sliders
        self.s_scale = QSlider()    # DOUBLED TO MAKE INT WORK
        self.s_scale.setRange(0,50)
        self.s_scale.setValue(0)
        self.s_scale.setOrientation(Qt.Horizontal)
        self.s_scale.setPageStep(4)
        self.s_sigma = QSlider()    # DOUBLED TO MAKE INT WORK
        self.s_sigma.setRange(1,20)
        self.s_sigma.setValue(1)
        self.s_sigma.setOrientation(Qt.Horizontal)
        self.s_sigma.setPageStep(2)
        self.s_gamma = QSlider() 
        self.s_gamma.setRange(1,1000)
        self.s_gamma.setValue(5)
        self.s_gamma.setOrientation(Qt.Horizontal)
        self.s_gamma.setPageStep(5)
        self.s_kernel_size = QSlider()
        self.s_kernel_size.setRange(1,10)
        self.s_kernel_size.setValue(1)
        self.s_kernel_size.setOrientation(Qt.Horizontal)
        self.s_kernel_size.setPageStep(2)
        self.s_min_thick = QSlider()    # DOUBLED TO MAKE INT WORK
        self.s_min_thick.setRange(2,10)
        self.s_min_thick.setValue(2)
        self.s_min_thick.setOrientation(Qt.Horizontal)
        self.s_min_thick.setPageStep(2)
        self.s_thin = QSlider()
        self.s_thin.setRange(1,5)
        self.s_thin.setValue(1)
        self.s_thin.setOrientation(Qt.Horizontal)
        self.s_thin.setPageStep(2)
        self.s_min_size = QSlider()
        self.s_min_size.setRange(1,200)
        self.s_min_size.setValue(1)
        self.s_min_size.setOrientation(Qt.Horizontal)
        self.s_min_size.setPageStep(10)
        self.s_max_hole_size = QSlider()
        self.s_max_hole_size.setRange(1,100)
        self.s_max_hole_size.setValue(10)
        self.s_max_hole_size.setOrientation(Qt.Horizontal)
        self.s_max_hole_size.setPageStep(2)

        # Numeric Labels
        self.n_scale = QLabel()
        self.n_scale.setText("0")
        self.n_sigma = QLabel()
        self.n_sigma.setText("0.5")
        self.n_gamma = QLabel()
        self.n_gamma.setText("5")
        self.n_kernel_size = QLabel()
        self.n_kernel_size.setText("1")
        self.n_min_thick = QLabel()
        self.n_min_thick.setText("1")
        self.n_thin = QLabel()
        self.n_thin.setText("1")
        self.n_min_size = QLabel()
        self.n_min_size.setText("1")
        self.n_max_hole_size = QLabel()
        self.n_max_hole_size.setText("10")

        # Link sliders and numeric labels
        self.s_scale.valueChanged.connect(self._update_scale)
        self.s_sigma.valueChanged.connect(self._update_sigma)
        self.s_gamma.valueChanged.connect(self._update_gamma)
        self.s_kernel_size.valueChanged.connect(self._update_kernel_size)
        self.s_min_thick.valueChanged.connect(self._update_min_thick)
        self.s_thin.valueChanged.connect(self._update_thin)
        self.s_min_size.valueChanged.connect(self._update_min_size)
        self.s_max_hole_size.valueChanged.connect(self._update_max_hole_size)

        # Buttons
        self.btn_preset = QPushButton("Run Preset")
        self.btn_smoothing = QPushButton("Run")
        self.btn_threshold = QPushButton("Run")
        self.btn_vesselness = QPushButton("Run")
        self.btn_merge = QPushButton("Run")
        self.btn_closing = QPushButton("Run")
        self.btn_thinning = QPushButton("Run")
        self.btn_cleaning = QPushButton("Run")
        self.btn_hole = QPushButton("Run")
        self.btn_skeleton = QPushButton("Run")

        # Add functions to buttons
        self.btn_preset.clicked.connect(self._run_preset)
        self.btn_smoothing.clicked.connect(self._smoothing)
        self.btn_threshold.clicked.connect(self._threshold)
        self.btn_vesselness.clicked.connect(self._vesselness)
        self.btn_merge.clicked.connect(self._merge)
        self.btn_closing.clicked.connect(self._closing)
        self.btn_thinning.clicked.connect(self._thinning)
        self.btn_cleaning.clicked.connect(self._cleaning)
        self.btn_hole.clicked.connect(self._hole_removal)
        self.btn_skeleton.clicked.connect(self._skeleton)

        # Horizontal lines
        self.line_1 = QWidget()
        self.line_1.setFixedHeight(4)
        self.line_1.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_1.setStyleSheet("background-color: #c0c0c0")
        self.line_2 = QWidget()
        self.line_2.setFixedHeight(2)
        self.line_2.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_2.setStyleSheet("background-color: #c0c0c0")
        self.line_3 = QWidget()
        self.line_3.setFixedHeight(4)
        self.line_3.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_3.setStyleSheet("background-color: #c0c0c0")
        self.line_4 = QWidget()
        self.line_4.setFixedHeight(4)
        self.line_4.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_4.setStyleSheet("background-color: #c0c0c0")
        self.line_5 = QWidget()
        self.line_5.setFixedHeight(4)
        self.line_5.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_5.setStyleSheet("background-color: #c0c0c0")
        self.line_6 = QWidget()
        self.line_6.setFixedHeight(4)
        self.line_6.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_6.setStyleSheet("background-color: #c0c0c0")
        self.line_7 = QWidget()
        self.line_7.setFixedHeight(4)
        self.line_7.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_7.setStyleSheet("background-color: #c0c0c0")
        self.line_8 = QWidget()
        self.line_8.setFixedHeight(4)
        self.line_8.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_8.setStyleSheet("background-color: #c0c0c0")

        # Combo boxes
        self.c_preset = QComboBox()
        self.c_preset.addItem("Bladder")
        self.c_preset.addItem("Bone")
        self.c_preset.addItem("Brain")
        self.c_preset.addItem("Ear")
        self.c_preset.addItem("Heart")
        self.c_preset.addItem("Liver")
        self.c_preset.addItem("Muscle")
        self.c_preset.addItem("Spinal Cord")
        self.c_preset.addItem("Tongue")
        self.c_preset_input = QComboBox()
        self.c_smoothing = QComboBox()
        self.c_threshold = QComboBox()
        self.c_vesselness = QComboBox()
        self.c_operation_dim = QComboBox()
        self.c_operation_dim.addItem("2D")
        self.c_operation_dim.addItem("3D")
        self.c_operation_dim.setCurrentIndex(1)
        self.c_cutoff_method = QComboBox()
        self.c_cutoff_method.addItem("threshold_li")
        self.c_cutoff_method.addItem("threshold_triangle")
        self.c_cutoff_method.addItem("threshold_otsu")
        self.c_merge_1 = QComboBox()
        self.c_merge_2 = QComboBox()
        self.c_merge_3 = QComboBox()
        self.c_closing = QComboBox()
        self.c_thinning = QComboBox()
        self.c_cleaning = QComboBox()
        self.c_hole = QComboBox()
        self.c_skeleton = QComboBox()
        self.list_comboboxes = [
            self.c_preset_input,
            self.c_smoothing,
            self.c_threshold,
            self.c_vesselness,
            self.c_merge_1,
            self.c_merge_2,
            self.c_merge_3,
            self.c_closing,
            self.c_thinning,
            self.c_cleaning,
            self.c_hole,
            self.c_skeleton
        ]

        # Add content to layer selecting comboboxes
        self._update_layer_lists()
        self.viewer.layers.events.inserted.connect(self._update_layer_lists)
        self.viewer.layers.events.removed.connect(self._update_layer_lists)
        self.viewer.layers.events.reordered.connect(self._update_layer_lists)
        self.viewer.layers.events.changed.connect(self._update_layer_lists)

        # Zone 0 (Preset Zone)

        self.h_0_1 = QWidget()
        self.h_0_1.setLayout(QHBoxLayout())
        self.h_0_1.layout().addWidget(self.l_preset_layer)
        self.h_0_1.layout().addWidget(self.c_preset_input)
        self.h_0_2 = QWidget()
        self.h_0_2.setLayout(QHBoxLayout())
        self.h_0_2.layout().addWidget(self.l_preset_name)
        self.h_0_2.layout().addWidget(self.c_preset)
        self.zone_0 = QWidget()
        self.zone_0.setLayout(QVBoxLayout())
        self.zone_0.layout().addWidget(self.h_0_1)
        self.zone_0.layout().addWidget(self.h_0_2)
        self.zone_0.layout().addWidget(self.btn_preset)

        # Zone 1
        self.h_1 = QWidget()
        self.h_1.setLayout(QHBoxLayout())
        self.h_1.layout().addWidget(self.l_1)
        self.h_1.layout().addWidget(self.c_smoothing)
        self.h_1.layout().addWidget(self.btn_smoothing)
        self.zone_1 = QWidget()
        self.zone_1.setLayout(QVBoxLayout())
        self.zone_1.layout().addWidget(self.h_1)

        # Zone 2
        self.h_2_1 = QWidget()
        self.h_2_1.setLayout(QHBoxLayout())
        self.h_2_1.layout().addWidget(self.l_2)
        self.h_2_1.layout().addWidget(self.c_threshold)
        self.h_2_1.layout().addWidget(self.btn_threshold)
        self.h_2_2 = QWidget()
        self.h_2_2.setLayout(QHBoxLayout())
        self.h_2_2.layout().addWidget(self.l_scale)
        self.h_2_2.layout().addWidget(self.s_scale)
        self.h_2_2.layout().addWidget(self.n_scale)
        self.zone_2 = QWidget()
        self.zone_2.setLayout(QVBoxLayout())
        self.zone_2.layout().addWidget(self.h_2_1)
        self.zone_2.layout().addWidget(self.h_2_2)

        # Zone 3
        self.h_3_1 = QWidget()
        self.h_3_1.setLayout(QHBoxLayout())
        self.h_3_1.layout().addWidget(self.l_3)
        self.h_3_1.layout().addWidget(self.c_vesselness)
        self.h_3_1.layout().addWidget(self.btn_vesselness)
        self.h_3_2 = QWidget()
        self.h_3_2.setLayout(QHBoxLayout())
        self.h_3_2.layout().addWidget(self.l_sigma)
        self.h_3_2.layout().addWidget(self.s_sigma)
        self.h_3_2.layout().addWidget(self.n_sigma)
        self.h_3_3 = QWidget()
        self.h_3_3.setLayout(QHBoxLayout())
        self.h_3_3.layout().addWidget(self.l_gamma)
        self.h_3_3.layout().addWidget(self.s_gamma)
        self.h_3_3.layout().addWidget(self.n_gamma)
        self.h_3_4 = QWidget()
        self.h_3_4.setLayout(QHBoxLayout())
        self.h_3_4.layout().addWidget(self.l_cutoff_method)
        self.h_3_4.layout().addWidget(self.c_cutoff_method)
        self.zone_3 = QWidget()
        self.zone_3.setLayout(QVBoxLayout())
        self.zone_3.layout().addWidget(self.h_3_1)
        self.zone_3.layout().addWidget(self.h_3_2)
        self.zone_3.layout().addWidget(self.h_3_3)
        self.zone_3.layout().addWidget(self.h_3_4)

        # Zone 4
        self.h_4_1 = QWidget()
        self.h_4_1.setLayout(QHBoxLayout())
        self.h_4_1.layout().addWidget(self.l_4)
        self.h_4_1.layout().addWidget(self.btn_merge)
        self.h_4_2 = QWidget()
        self.h_4_2.setLayout(QHBoxLayout())
        self.h_4_2.layout().addWidget(self.l_threshold_result)
        self.h_4_2.layout().addWidget(self.c_merge_1)
        self.h_4_3 = QWidget()
        self.h_4_3.setLayout(QHBoxLayout())
        self.h_4_3.layout().addWidget(self.l_vesselness_1)
        self.h_4_3.layout().addWidget(self.c_merge_2)
        self.h_4_4 = QWidget()
        self.h_4_4.setLayout(QHBoxLayout())
        self.h_4_4.layout().addWidget(self.l_vesselness_2)
        self.h_4_4.layout().addWidget(self.c_merge_3)
        self.zone_4 = QWidget()
        self.zone_4.setLayout(QVBoxLayout())
        self.zone_4.layout().addWidget(self.h_4_1)
        self.zone_4.layout().addWidget(self.h_4_2)
        self.zone_4.layout().addWidget(self.h_4_3)
        self.zone_4.layout().addWidget(self.h_4_4)

        # Zone 5
        self.h_5_1 = QWidget()
        self.h_5_1.setLayout(QHBoxLayout())
        self.h_5_1.layout().addWidget(self.l_5)
        self.h_5_1.layout().addWidget(self.c_closing)
        self.h_5_1.layout().addWidget(self.btn_closing)
        self.h_5_2 = QWidget()
        self.h_5_2.setLayout(QHBoxLayout())
        self.h_5_2.layout().addWidget(self.l_kernel_size)
        self.h_5_2.layout().addWidget(self.s_kernel_size)
        self.h_5_2.layout().addWidget(self.n_kernel_size)
        self.zone_5 = QWidget()
        self.zone_5.setLayout(QVBoxLayout())
        self.zone_5.layout().addWidget(self.h_5_1)
        self.zone_5.layout().addWidget(self.h_5_2)

        # Zone 6
        self.h_6_1 = QWidget()
        self.h_6_1.setLayout(QHBoxLayout())
        self.h_6_1.layout().addWidget(self.l_6)
        self.h_6_1.layout().addWidget(self.c_thinning)
        self.h_6_1.layout().addWidget(self.btn_thinning)
        self.h_6_2 = QWidget()
        self.h_6_2.setLayout(QHBoxLayout())
        self.h_6_2.layout().addWidget(self.l_min_thick)
        self.h_6_2.layout().addWidget(self.s_min_thick)
        self.h_6_2.layout().addWidget(self.n_min_thick)
        self.h_6_3 = QWidget()
        self.h_6_3.setLayout(QHBoxLayout())
        self.h_6_3.layout().addWidget(self.l_thin)
        self.h_6_3.layout().addWidget(self.s_thin)
        self.h_6_3.layout().addWidget(self.n_thin)
        self.zone_6 = QWidget()
        self.zone_6.setLayout(QVBoxLayout())
        self.zone_6.layout().addWidget(self.h_6_1)
        self.zone_6.layout().addWidget(self.h_6_2)
        self.zone_6.layout().addWidget(self.h_6_3)

        # Zone 7
        self.h_7_1 = QWidget()
        self.h_7_1.setLayout(QHBoxLayout())
        self.h_7_1.layout().addWidget(self.l_7)
        self.h_7_1.layout().addWidget(self.c_cleaning)
        self.h_7_1.layout().addWidget(self.btn_cleaning)
        self.h_7_2 = QWidget()
        self.h_7_2.setLayout(QHBoxLayout())
        self.h_7_2.layout().addWidget(self.l_min_size)
        self.h_7_2.layout().addWidget(self.s_min_size)
        self.h_7_2.layout().addWidget(self.n_min_size)
        self.zone_7 = QWidget()
        self.zone_7.setLayout(QVBoxLayout())
        self.zone_7.layout().addWidget(self.h_7_1)
        self.zone_7.layout().addWidget(self.h_7_2)

        # Zone 8
        self.h_8_1 = QWidget()
        self.h_8_1.setLayout(QHBoxLayout())
        self.h_8_1.layout().addWidget(self.l_8)
        self.h_8_1.layout().addWidget(self.c_hole)
        self.h_8_1.layout().addWidget(self.btn_hole)
        self.h_8_2 = QWidget()
        self.h_8_2.setLayout(QHBoxLayout())
        self.h_8_2.layout().addWidget(self.l_max_hole_size)
        self.h_8_2.layout().addWidget(self.s_max_hole_size)
        self.h_8_2.layout().addWidget(self.n_max_hole_size)
        self.zone_8 = QWidget()
        self.zone_8.setLayout(QVBoxLayout())
        self.zone_8.layout().addWidget(self.h_8_1)
        self.zone_8.layout().addWidget(self.h_8_2)

        # Zone 9
        self.h_9_1 = QWidget()
        self.h_9_1.setLayout(QHBoxLayout())
        self.h_9_1.layout().addWidget(self.l_9)
        self.h_9_1.layout().addWidget(self.c_skeleton)
        self.h_9_1.layout().addWidget(self.btn_skeleton)
        self.zone_9 = QWidget()
        self.zone_9.setLayout(QVBoxLayout())
        self.zone_9.layout().addWidget(self.h_9_1)

        # Layouting
        self.content = QWidget()
        self.content.setLayout(QVBoxLayout())
        self.content.layout().addWidget(self.zone_0)
        self.content.layout().addWidget(self.l_title)
        self.content.layout().addWidget(self.zone_1)
        self.content.layout().addWidget(self.line_1)
        self.content.layout().addWidget(self.zone_2)
        self.content.layout().addWidget(self.line_2)
        self.content.layout().addWidget(self.zone_3)
        self.content.layout().addWidget(self.line_3)
        self.content.layout().addWidget(self.zone_4)
        self.content.layout().addWidget(self.line_4)
        self.content.layout().addWidget(self.zone_5)
        self.content.layout().addWidget(self.line_5)
        self.content.layout().addWidget(self.zone_6)
        self.content.layout().addWidget(self.line_6)
        self.content.layout().addWidget(self.zone_7)
        self.content.layout().addWidget(self.line_7)
        self.content.layout().addWidget(self.zone_8)
        self.content.layout().addWidget(self.line_8)
        self.content.layout().addWidget(self.zone_9)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.content)

        # Setting layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.scroll_area)

    # Slider update functions
    def _update_scale(self):
        self.n_scale.setText(str(self.s_scale.value()/2))

    def _update_sigma(self):
        self.n_sigma.setText(str(self.s_sigma.value()/2))

    def _update_gamma(self):
        self.n_gamma.setText(str(self.s_gamma.value()))

    def _update_kernel_size(self):
        self.n_kernel_size.setText(str(self.s_kernel_size.value()))

    def _update_min_thick(self):
        self.n_min_thick.setText(str(self.s_min_thick.value()/2))

    def _update_thin(self):
        self.n_thin.setText(str(self.s_thin.value()))

    def _update_min_size(self):
        self.n_min_size.setText(str(self.s_min_size.value()))

    def _update_max_hole_size(self):
        self.n_max_hole_size.setText(str(self.s_max_hole_size.value()))

    # Button onclick functions
    def _smoothing(self, preset = False, data = ""):
        """
        perform edge preserving smoothing
        """

        if not preset:
            selected_layer = self.c_smoothing.currentText()
            for layer in self.viewer.layers:
                if layer.name == selected_layer and type(layer) == Image:
                    data = layer.data
                    break
        out = edge_preserving_smoothing_3d(data)
        self.viewer.add_image(data = out, name = "smoothed_Image")
        if preset:
            return out

    def _threshold(self, preset = False, image = "", scale = 0):   # HALVE VALUE
        """
        apply vesselness filter on images
        Parameters:
        -------------
        image: np.ndarray
            the image to be applied on
        scale: Union[float, int]
            how many fold of the standard deviation of the image intensity 
            will be used to calculate the threshold
        Return
        -------------
        np.ndarray
        """

        if not preset:
            selected_layer = self.c_smoothing.currentText()
            for layer in self.viewer.layers:
                if layer.name == selected_layer and type(layer) == Image:
                    image = layer.data
                    break
            scale = self.s_scale.value()/2
        thresh = image.mean() + scale * image.std()
        out = image > thresh
        self.viewer.add_image(data = out, name = f"threshold_{scale}", blending="additive")
        if preset:
            return out

    def _vesselness(self, preset = False, image = "", sigma = 0, gamma = 5, dim = 3, cutoff_method = ""):  # HALVE VALUE
        """
        apply vesselness filter on images
        Parameters:
        -------------
        image: np.ndarray
            the image to be applied on
        dim: int
            the dimenstion of the operation, 2 or 3
        sigma: float
            the kernal size of the vesselness filter
        cutoff_method: str
            the method to use for binarization
        Return
        -------------
        np.ndarray
        """

        if not preset:
            selected_layer = self.c_vesselness.currentText()
            for layer in self.viewer.layers:
                if layer.name == selected_layer and type(layer) == Image:
                    image = layer.data
                    break
            dim = 3 #[2,3][(self.c_operation_dim.currentText() == "3D")]
            sigma = self.s_sigma.value()/2
            gamma = self.s_gamma.value()
            cutoff_method = self.c_cutoff_method.currentText()
        out = vesselness_filter(image, dim, sigma, gamma, cutoff_method)
        self.viewer.add_image(data = out, name = f"ves_{sigma}_{gamma}_{cutoff_method}", blending="additive")
        if preset:
            return out

    def _merge(self, preset = False, layers = 0, data1 = "", data2 = "", data3 = ""):
        if not preset:
            layer_list = [
                self.c_merge_1.currentText(),
                self.c_merge_2.currentText(),
                self.c_merge_3.currentText()
            ]
            counter = 0
            for layer in self.viewer.layers:
                if layer.name in layer_list and type(layer) == Image:
                    image = layer.data
                    if counter == 0:
                        seg = image > 0
                    else:
                        seg = np.logical_or(seg, image > 0)
                    counter += 1
                    if counter == 3:
                        break
        else:
            seg = data1 > 0
            seg = np.logical_or(seg, data2 > 0)
            if layers == 3:
                seg = np.logical_or(seg, data3 > 0)
        self.viewer.add_image(data = seg, name = "merged_segmentation", blending="additive")
        if preset:
            return seg
        

    def _closing(self, preset = False, image = "", kernel = 0):
        """
        perform morphological closing to remove small gaps in segmentation
        Parameters:
        -------------
        image: np.ndarray
            the image to be applied on
        scale: int
            the kernal size of the closing operation
        Return
        -------------
        np.ndarray
        """

        if not preset:
            selected_layer = self.c_closing.currentText()
            for layer in self.viewer.layers:
                if layer.name == selected_layer and type(layer) == Image:
                    image = layer.data
                    break
            kernel = self.s_kernel_size.value()
        out = binary_closing(image, cube(kernel))
        self.viewer.add_image(data = out, name = f"closing_{kernel}", blending="additive")
        if preset:
            return out
    
    def _hole_removal(self, preset = False, image = "", max_size = 0):
        """
        remove small holes in segmentation
        Parameters:
        -------------
        image: np.ndarray
            the image to be applied on
        max_size: int
            the max hole size to remove
        Return
        -------------
        np.ndarray
        """
        from aicssegmentation.core.utils import hole_filling
        if not preset:
            selected_layer = self.c_hole.currentText()
            for layer in self.viewer.layers:
                if layer.name == selected_layer and type(layer) == Image:
                    image = layer.data
                    break
            max_size = self.s_max_hole_size.value()
        out = hole_filling(image, hole_min=1, hole_max=max_size, fill_2d=True)
        self.viewer.add_image(data = out, name = f"hole_filled_{max_size}", blending="additive")
        if preset:
            return out

    def _thinning(self, preset = False, image ="", min_thickness = 0, thin = 0):    # HALVE ONE VALUE
        """
        perform topology preserving thinning
        Parameters:
        -------------
        image: np.ndarray
            the image to be applied on
        min_thickness: float
            the minimal thickness to kept without breaking
        thin: int
            the amount of thinning
        Return
        -------------
        np.ndarray
        """

        if not preset:
            selected_layer = self.c_thinning.currentText()
            for layer in self.viewer.layers:
                if layer.name == selected_layer and type(layer) == Image:
                    image = layer.data
                    break
            min_thickness = self.s_min_thick.value()/2
            thin = self.s_thin.value()
        out = topology_preserving_thinning(image > 0, min_thickness, thin)

        self.viewer.add_image(data = out, name = f"thinned_{min_thickness}_{thin}", blending="additive")
        if preset:
            return out

    def _cleaning(self, preset = False, image = "", min_size = 0):
        """
        clean up small objects from the segmentation result
        Parameters:
        -------------
        image: np.ndarray
            the image to be applied on
        min_size: int
            the size for objects to be cleaned
        Return
        -------------
        np.ndarray
        """

        if not preset:
            selected_layer = self.c_cleaning.currentText()
            for layer in self.viewer.layers:
                if layer.name == selected_layer and type(layer) == Image:
                    image = layer.data
                    break
            min_size = self.s_min_size.value()
        out = remove_small_objects(image > 0, min_size)
        self.viewer.add_image(data = out, name = f"cleaned_{min_size}", blending="additive")

    def _skeleton(self, preset = False, image =""):    # HALVE ONE VALUE
        """
        perform skeletonization

        Parameters:
        -------------
        image: np.ndarray
            the image to be applied on

        Return
        -------------
        np.ndarray
        """
        from skimage.morphology import skeletonize_3d
        if not preset:
            selected_layer = self.c_skeleton.currentText()
            for layer in self.viewer.layers:
                if layer.name == selected_layer and type(layer) == Image:
                    image = layer.data
                    break
        out = skeletonize_3d(image > 0)

        self.viewer.add_image(data = out, name = "skeleton", blending="additive")
        if preset:
            return out

    # Combobox update function
    def _update_layer_lists(self, index = 0, new_index = 0, old_value = "", value = "", ):
        for box in self.list_comboboxes:
            box.clear()
            if box == self.c_merge_3:
                box.addItem("N/A")
        names = []
        for layer in self.viewer.layers:
            if type(layer) == Image:
                names.append(layer.name)
        for box in self.list_comboboxes:
            for name in names:
                box.addItem(name)

    # Preset function
    def _run_preset(self):
        """
        runs the selected preset on the selected layer without interaction from the user necessary
        """

        selected_layer = self.c_preset_input.currentText()
        for layer in self.viewer.layers:
            if layer.name == selected_layer and type(layer) == Image:
                image = layer.data
                break

        if self.c_preset.currentIndex() == 0: # Bladder preset
            smooth_image = self._smoothing(preset = True, data = image)
            vessel1 = self._threshold(preset = True, image = smooth_image, scale = 3)
            vessel2 = self._vesselness(preset = True, image = smooth_image, sigma = 1, gamma=5, cutoff_method = "threshold_triangle")
            vessel3 = self._vesselness(preset = True, image = smooth_image, sigma = 3, gamma=5, cutoff_method = "threshold_otsu")
            merge = self._merge(preset = True, layers = 3, data1 = vessel1, data2 = vessel2, data3 = vessel3)
            closed = self._closing(preset = True, image = merge, kernel = 5)
            thinned = self._thinning(preset = True, image = closed, min_thickness = 1, thin = 1)
            self._cleaning(preset = True, image = thinned, min_size = 100)

        elif self.c_preset.currentIndex() == 1: # bone preset
            smooth_image = self._smoothing(preset = True, data = image)
            vessel1 = self._threshold(preset = True, image = smooth_image, scale = 3)
            vessel2 = self._vesselness(preset = True, image = smooth_image, sigma = 1, gamma = 110, cutoff_method = "threshold_li")
            merge = self._merge(preset = True, layers = 2, data1 = vessel1, data2 = vessel2)
            closed = self._closing(preset = True, image = merge, kernel = 3)
            self._cleaning(preset = True, image = closed, min_size = 100)

        elif self.c_preset.currentIndex() == 2: # Brain preset
            smooth_image = self._smoothing(preset = True, data = image)
            vessel1 = self._threshold(preset = True, image = smooth_image, scale = 3)
            vessel2 = self._vesselness(preset = True, image = smooth_image, sigma = 1, gamma = 5, cutoff_method = "threshold_li")
            vessel3 = self._vesselness(preset = True, image = smooth_image, sigma = 2, gamma = 5, cutoff_method = "threshold_li")
            merge = self._merge(preset = True, layers = 3, data1 = vessel1, data2 = vessel2, data3 = vessel3)
            closed = self._closing(preset = True, image = merge, kernel = 5)
            self._cleaning(preset = True, image = closed, min_size = 100)

        elif self.c_preset.currentIndex() == 3: # Ear preset
            smooth_image = self._smoothing(preset = True, data = image)
            vessel1 = self._threshold(preset = True, image = smooth_image, scale = 2)
            vessel2 = self._vesselness(preset = True, image = smooth_image, sigma = 1, gamma = 120, cutoff_method = "threshold_triangle")
            vessel3 = self._vesselness(preset = True, image = smooth_image, sigma = 2, gamma = 120, cutoff_method = "threshold_li")
            merge = self._merge(preset = True, layers = 3, data1 = vessel1, data2 = vessel2, data3 = vessel3)
            closed = self._closing(preset = True, image = merge, kernel = 3)
            thinned = self._thinning(preset = True, image = closed, min_thickness = 1, thin = 1)
            self._cleaning(preset = True, image = thinned, min_size = 20)

        elif self.c_preset.currentIndex() == 4: # Heart preset
            smooth_image = self._smoothing(preset = True, data = image)
            vessel1 = self._threshold(preset = True, image = smooth_image, scale = 3)
            vessel2 = self._vesselness(preset = True, image = smooth_image, sigma = 1, gamma = 5, cutoff_method = "threshold_li")
            vessel3 = self._vesselness(preset = True, image = smooth_image, sigma = 2, gamma = 5, cutoff_method = "threshold_otsu")
            merge = self._merge(preset = True, layers = 3, data1 = vessel1, data2 = vessel2, data3 = vessel3)
            thinned = self._thinning(preset = True, image = merge, min_thickness = 1, thin = 1)
            self._cleaning(preset = True, image = thinned, min_size = 100)

        elif self.c_preset.currentIndex() == 5: # Liver preset
            smooth_image = self._smoothing(preset = True, data = image)
            vessel1 = self._threshold(preset = True, image = smooth_image, scale = 3)
            vessel2 = self._vesselness(preset = True, image = smooth_image, sigma = 2, gamma = 10, cutoff_method = "threshold_li")
            merge = self._merge(preset = True, layers = 2, data1 = vessel1, data2 = vessel2)
            closed = self._closing(preset = True, image = merge, kernel = 5)
            self._cleaning(preset = True, image = closed, min_size = 100)

        elif self.c_preset.currentIndex() == 6: # Muscle preset
            smooth_image = self._smoothing(preset = True, data = image)
            vessel1 = self._threshold(preset = True, image = smooth_image, scale = 3.5)
            vessel2 = self._vesselness(preset = True, image = smooth_image, sigma = 1, gamma = 70, cutoff_method = "threshold_triangle")
            vessel3 = self._vesselness(preset = True, image = smooth_image, sigma = 2, gamma = 90, cutoff_method = "threshold_li")
            merge = self._merge(preset = True, layers = 3, data1 = vessel1, data2 = vessel2, data3 = vessel3)
            self._cleaning(preset = True, image = merge, min_size = 20)

        elif self.c_preset.currentIndex() == 7: # Spinal cord preset
            smooth_image = self._smoothing(preset = True, data = image)
            vessel1 = self._threshold(preset = True, image = smooth_image, scale = 3)
            vessel2 = self._vesselness(preset = True, image = smooth_image, sigma = 1, gamma = 5, cutoff_method = "threshold_triangle")
            vessel3 = self._vesselness(preset = True, image = smooth_image, sigma = 2, gamma = 5, cutoff_method = "threshold_triangle")
            merge = self._merge(preset = True, layers = 3, data1 = vessel1, data2 = vessel2, data3 = vessel3)
            closed = self._closing(preset = True, image = merge, kernel = 3)
            thinned = self._thinning(preset = True, image = closed, min_thickness = 1, thin = 1)
            self._cleaning(preset = True, image = thinned, min_size = 100)

        elif self.c_preset.currentIndex() == 8: # Tongue preset
            smooth_image = self._smoothing(preset = True, data = image)
            vessel1 = self._threshold(preset = True, image = smooth_image, scale = 4)
            vessel2 = self._vesselness(preset = True, image = smooth_image, sigma = 1, gamma = 170, cutoff_method = "threshold_triangle")
            vessel3 = self._vesselness(preset = True, image = smooth_image, sigma = 1, gamma = 40, cutoff_method = "threshold_li")
            merge = self._merge(preset = True, layers = 3, data1 = vessel1, data2 = vessel2, data3=vessel3)
            closed = self._closing(preset = True, image = merge, kernel = 3)
            thinned = self._thinning(preset = True, image = closed, min_thickness = 1, thin = 1)
            self._cleaning(preset = True, image = thinned, min_size = 20)

    """
    # This can be interesting if we decide to use the currently selected layers instead of comboboxes
    def _get_data_from_layer(self, amount = 1):
        data = []
        # TODO: Check if layerlist has as many imagelayers as is requested
        for layer in self.viewer.layers.selection:
            if type(layer) == Image:
                data.append(layer.data)
                amount = amount - 1
                if amount == 0:
                    return data
        # TODO: Implement error if less layers exist than requested
        pass
    """


#################################
# the second part
#################################
class Evaluation(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.l_title = QLabel("<font color='green'>evaluate-segmentation:</font>")
        self.l_evaluate = QLabel("Please evaluate the segmentation!")
        self.l_directory = QLabel("No file displayed yets")


        self.btn_next = QPushButton("Next image")
        self.btn_save = QPushButton("Save")
        self.btn_dir = QPushButton("Choose a directory")
        self.btn_next.clicked.connect(self._next)
        self.btn_save.clicked.connect(self._save)
        self.btn_dir.clicked.connect(self._select_dir)

        
        self.c_eval = QComboBox()
        self.c_eval.addItem("No evaluation")
        self.c_eval.addItem("Good")
        self.c_eval.addItem("Failed Segmentation")
        self.c_eval.addItem("Bad Image")

        self.line_1 = QWidget()
        self.line_1.setFixedHeight(4)
        self.line_1.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_1.setStyleSheet("background-color: #c0c0c0")

        self.line_2 = QWidget()
        self.line_2.setFixedHeight(4)
        self.line_2.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_2.setStyleSheet("background-color: #c0c0c0")
        
        # Layouting
        self.content = QWidget()
        self.content.setLayout(QVBoxLayout())
        self.content.layout().addWidget(self.l_title)
        self.content.layout().addWidget(self.btn_dir)
        self.content.layout().addWidget(self.line_1)
        self.content.layout().addWidget(self.l_directory)
        self.content.layout().addWidget(self.btn_next)
        self.content.layout().addWidget(self.l_evaluate)
        self.content.layout().addWidget(self.c_eval)
        self.content.layout().addWidget(self.line_2)
        self.content.layout().addWidget(self.btn_save)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.content)

        # Setting layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.scroll_area)

        self.evaluated = []

    def _select_dir(self):
        self.directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.filenames = glob(self.directory + os.sep + "Binary_*.tiff")
        self.filenames.sort()
        self.current_fn = None
        self.total_num = len(self.filenames)
        self._next()

    def _next(self): # Removes all current layers and loads in two corresponding images as layers
        if len(self.filenames) == 0:
            noMoreFiles = QMessageBox()
            noMoreFiles.setText("There are no more file tuples to evaluate!")
            noMoreFiles.exec()
            return
        bw_fn = self.filenames.pop() # Assume the file name is Binary_xxxx.tiff
        raw_fn = self.directory + os.sep + os.path.basename(bw_fn)[7:]
        if not os.path.exists(raw_fn):
            raw_fn = raw_fn[:-1]  # .tiff --> .tif
        self.l_directory.setText(f"{self.total_num-len(self.filenames)} / {self.total_num}")
        original = imread(raw_fn)
        segmentation = imread(bw_fn)
        self._remove_layers()
        if self.current_fn is not None:
            self._eval()
        vlow = np.percentile(original, 0.5)
        vhigh = np.percentile(original, 99.5)
        self.viewer.add_image(data = original, name = "Original data", blending = "additive", contrast_limits=[vlow, vhigh])
        self.viewer.add_image(data = segmentation, name = "Segmentation data", colormap="magenta", blending = "additive")
        self.current_fn = raw_fn
        

    def _eval(self):
        self.evaluated.append(os.path.basename(self.current_fn) + ", " + self.c_eval.currentText() + "\n")

    def _save(self):
        if self.current_fn is not None:
            self._eval()
        filename = QFileDialog.getSaveFileName(self, caption = "test", directory  = self.directory, filter = "*.csv")
        outfile = open(filename[0],"a")
        for i in range(0,len(self.evaluated)):
            outfile.write(self.evaluated[i])
        outfile.close()
        msgBox = QMessageBox()
        msgBox.setText("The file has been saved.")
        msgBox.exec()

    def _remove_layers(self):
        rm = []
        for layer in self.viewer.layers: # Split into two loops because one loop ignors some layers
            rm.append(layer.name)
        for layer in rm:
            self.viewer.layers.remove(layer)
        

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [ParameterTuning, Evaluation]