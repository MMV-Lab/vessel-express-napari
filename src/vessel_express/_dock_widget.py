from PyQt5.QtWidgets import QComboBox, QLabel, QSizePolicy
import napari
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QPushButton, QSlider, QHBoxLayout, QVBoxLayout, QScrollArea
from qtpy.QtCore import Qt
from napari.layers import Image

# packages required by processing functions
from .utils import vesselness_filter
from numpy import logical_or
from aicssegmentation.core.pre_processing_utils import  edge_preserving_smoothing_3d
from skimage.morphology import remove_small_objects, binary_closing, cube
from aicssegmentation.core.utils import topology_preserving_thinning


class VesselExpress(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Labels
        self.l_title = QLabel("<font color='green'>VesselExpress Segmentation Parameter Tuning:</font>")
        self.l_1 = QLabel("smoothing:")
        self.l_2 = QLabel("core-threshold:")
        self.l_3 = QLabel("core-vesselness:")
        self.l_4 = QLabel("merge segmentation:")
        self.l_5 = QLabel("post-closing:")
        self.l_6 = QLabel("post-thinning:")
        self.l_7 = QLabel("post-cleaning:")
        self.l_scale = QLabel("scale")
        self.l_sigma = QLabel("- sigma")
        self.l_operation_dim = QLabel("- operation_dim")
        self.l_cutoff_method = QLabel("- cutoff-method")
        self.l_threshold_result = QLabel("threshold result")
        self.l_vesselness_1 = QLabel("vesselness 1")
        self.l_vesselness_2 = QLabel("vesselness 2")
        self.l_kernel_size = QLabel("kernel-size")
        self.l_min_thick = QLabel("min_thickness")
        self.l_thin = QLabel("thin")
        self.l_min_size = QLabel("min_size")

        # Set tooltips
        smoothing_layer_tip = (
            "[Pre-processing]: Smooth your image without losing sharp edges.<br>\n"
            "You may not need this step if (1) your image has little noise<br>\n"
            "and (2) the segmentation results are accurate enough.<br><br>\n \n"
            "Parameters: \n"
            "\t<p style='margin-left: 40px'>No parameters are needed.</p>\n \n"
            "Instruction: \n"
            "\t<p style='margin-left: 40px'>Select the image to apply the smoothing on and click \"Run\".</p>"
        )
        self.l_1.setToolTip(smoothing_layer_tip)
        core_thresh_tip = (
            "[Core segmentation step 1]: Extract the vessels with very high intensity. <br>\n"
            "The threshold is set as intensity_mean + 'scale' x intensity_standard_deviation. <br><br>\n \n"
            "Parameters: \n"
            "\t<p style='margin-left: 40px'>scale: A large value will result in a higher threshold. </p>\n \n"
            "Instruction: \n"
            "\t<p style='margin-left: 40px'>Select the image (usually the smoothed image) to apply on, select the scale and click \"Run\".</p>"
        )
        self.l_2.setToolTip(core_thresh_tip)
        core_vessel_tip = (
            "[Core segmentation step 2]: Use vesselness filter to extract filamentous objects.<br>\n"
            "A Frangi filter will be applied first and then a cutoff value will be calculated<br>\n"
            "to binarize the result.<br><br>\n\n"
            "<b>** Note**:</b> You can combine the results of two different vesselness filters in the merge step.<br>\n"
            "For example, you can use one filter to catch thinner vessels and use another filter to pick up<br>\n"
            "thicker vessels.<br><br>\n\n"
            "Parameters: \n"
            "\t<p style='margin-left: 40px'>sigma: The kernel size of the Frangi filter. Use a large sigma for thicker vessels.<br>\n"
            "\toperation_dim: Select weather to apply the filter in 3D or in 2D in a slice-by-slice manner.<br>\n"
            "\tcutoff-method: The method used to determine how to binarize the filter result as segmentation.</p>\n\n"
            "Instruction: \n"
            "\t<p style='margin-left: 40px'>Select the image (usually the smoothed image) to apply on, select parameters and click \"Run\".</p>"
        )
        self.l_3.setToolTip(core_vessel_tip)
        core_merge_tip = (
            "[Merging]: In most cases none of the above segmentation will work perfectly as a single filter.<br>\n"
            "We find that combining the thresholding result and up to two different vesselness filters can<br>\n"
            "yield good results for your applications.<br><br>\n \n"
            "Instruction: \n"
            "\t<p style='margin-left: 40px'>Select which version of the thresholding based method to use and select<br>\n"
            "\tup to two vesselness segmentation restuls, then click \"Run\" to see how the<br>\n"
            "\tresult looks after merging them.</p>"
        )
        self.l_4.setToolTip(core_merge_tip)
        post_clost_tip = (
            "[Post-processing 1]: The vesselness filter may have broken segmentation near junction areas.<br>\n"
            "You can use the closing step to bridge such gaps.<br><br>\n \n"
            "Parameters: \n"
            "\t<p style='margin-left: 40px'>kernel_size: A large value will close larger gaps, but may falsely merge proximal vessels.</p>\n\n"
            "Instruction: \n"
            "\t<p style='margin-left: 40px'>Select the segmentation result (usually after merging) to apply on,<br>\n"
            "\tselect the kernel_size and click \"Run\".</p>"
        )
        self.l_5.setToolTip(post_clost_tip)
        post_thin_tip = (
            "[Post-processing 2]: The segmentation result may look thicker than it should be due to<br>\n"
            "the diffraction of light. You can perform a thinning step without altering the topology.<br><br>\n\n"
            "Parameters: \n"
            "<p style='margin-left: 40px'>\tmin_thickness: Any vessel thinner than this value will <b>not</b> be further thinned.<br>\n"
            "\tthin: How many pixels to thin your segmentations by.</p>"
        )
        self.l_6.setToolTip(post_thin_tip)
        self.l_7.setToolTip("Any segmented objects smaller than min_size will be removed to clean up your result.")
        
        core_thresh_scale_tip = (
            "Larger value will result in higher threshold value,\n"
            "and therefore less pixels will be segmented."
        )
        self.l_scale.setToolTip(core_thresh_scale_tip)
        core_vessel_sigma_tip = (
            "Larger value will help you pick up thicker vessels."
        )
        self.l_sigma.setToolTip(core_vessel_sigma_tip)
        core_vessel_dim_tip = (
            "Choose 3D (apply 3D filter) or 2D (apply 2D filter slice by slice)."
        )
        self.l_operation_dim.setToolTip(core_vessel_dim_tip)
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
        self.s_kernel_size = QSlider()
        self.s_kernel_size.setRange(0,10)
        self.s_kernel_size.setValue(0)
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

        # Numeric Labels
        self.n_scale = QLabel()
        self.n_scale.setText("0")
        self.n_sigma = QLabel()
        self.n_sigma.setText("0.5")
        self.n_kernel_size = QLabel()
        self.n_kernel_size.setText("0")
        self.n_min_thick = QLabel()
        self.n_min_thick.setText("1")
        self.n_thin = QLabel()
        self.n_thin.setText("1")
        self.n_min_size = QLabel()
        self.n_min_size.setText("1")

        # Link sliders and numeric labels
        self.s_scale.valueChanged.connect(self._update_scale)
        self.s_sigma.valueChanged.connect(self._update_sigma)
        self.s_kernel_size.valueChanged.connect(self._update_kernel_size)
        self.s_min_thick.valueChanged.connect(self._update_min_thick)
        self.s_thin.valueChanged.connect(self._update_thin)
        self.s_min_size.valueChanged.connect(self._update_min_size)

        # Buttons
        self.btn_smoothing = QPushButton("Run")
        self.btn_threshold = QPushButton("Run")
        self.btn_vesselness = QPushButton("Run")
        self.btn_merge = QPushButton("Run")
        self.btn_closing = QPushButton("Run")
        self.btn_thinning = QPushButton("Run")
        self.btn_cleaning = QPushButton("Run")

        # Add functions to buttons
        self.btn_smoothing.clicked.connect(self._smoothing)
        self.btn_threshold.clicked.connect(self._threshold)
        self.btn_vesselness.clicked.connect(self._vesselness)
        self.btn_merge.clicked.connect(self._merge)
        self.btn_closing.clicked.connect(self._closing)
        self.btn_thinning.clicked.connect(self._thinning)
        self.btn_cleaning.clicked.connect(self._cleaning)

        # Horizontal lines
        self.line_1 = QWidget()
        self.line_1.setFixedHeight(2)
        self.line_1.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_1.setStyleSheet("background-color: #c0c0c0")
        self.line_2 = QWidget()
        self.line_2.setFixedHeight(2)
        self.line_2.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_2.setStyleSheet("background-color: #c0c0c0")
        self.line_3 = QWidget()
        self.line_3.setFixedHeight(2)
        self.line_3.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_3.setStyleSheet("background-color: #c0c0c0")
        self.line_4 = QWidget()
        self.line_4.setFixedHeight(2)
        self.line_4.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_4.setStyleSheet("background-color: #c0c0c0")
        self.line_5 = QWidget()
        self.line_5.setFixedHeight(2)
        self.line_5.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_5.setStyleSheet("background-color: #c0c0c0")
        self.line_6 = QWidget()
        self.line_6.setFixedHeight(2)
        self.line_6.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.line_6.setStyleSheet("background-color: #c0c0c0")

        # Combo boxes
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
        self.list_comboboxes = [self.c_smoothing,self.c_threshold,self.c_vesselness,self.c_merge_1,
            self.c_merge_2,self.c_merge_3,self.c_closing,self.c_thinning,self.c_cleaning]

        # Add content to layer selecting comboboxes
        self._update_layer_lists()
        self.viewer.layers.events.inserted.connect(self._update_layer_lists)
        self.viewer.layers.events.removed.connect(self._update_layer_lists)
        self.viewer.layers.events.reordered.connect(self._update_layer_lists)
        self.viewer.layers.events.changed.connect(self._update_layer_lists)

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
        self.h_3_3.layout().addWidget(self.l_operation_dim)
        self.h_3_3.layout().addWidget(self.c_operation_dim)
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

        # Layouting
        self.content = QWidget()
        self.content.setLayout(QVBoxLayout())
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

    def _update_kernel_size(self):
        self.n_kernel_size.setText(str(self.s_kernel_size.value()))

    def _update_min_thick(self):
        self.n_min_thick.setText(str(self.s_min_thick.value()/2))

    def _update_thin(self):
        self.n_thin.setText(str(self.s_thin.value()))

    def _update_min_size(self):
        self.n_min_size.setText(str(self.s_min_size.value()))

    # Button onclick functions
    def _smoothing(self):
        """
        perform edge preserving smoothing
        """

        selected_layer = self.c_smoothing.currentText()
        for layer in self.viewer.layers:
            if layer.name == selected_layer and type(layer) == Image:
                data = layer.data
                break
        out = edge_preserving_smoothing_3d(data)
        self.viewer.add_image(data = out, name = "smoothed_Image")

    def _threshold(self):   # HALVE VALUE
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

        selected_layer = self.c_smoothing.currentText()
        for layer in self.viewer.layers:
            if layer.name == selected_layer and type(layer) == Image:
                image = layer.data
                break
        scale = self.s_scale.value()/2
        thresh = image.mean() + scale * image.std()
        out = image > thresh
        self.viewer.add_image(data = out, name = f"threshold_{scale}", blending="additive")

    def _vesselness(self):  # HALVE VALUE
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

        selected_layer = self.c_smoothing.currentText()
        for layer in self.viewer.layers:
            if layer.name == selected_layer and type(layer) == Image:
                image = layer.data
                break
        dim = [2,3][(self.c_operation_dim.currentText() == "3D")]
        print(f"running {dim}D vesselness filter ...")
        sigma = self.s_sigma.value()/2
        cutoff_method = self.c_cutoff_method.currentText()
        out = vesselness_filter(image, dim, sigma, cutoff_method)
        print("vesselness filter is done")
        self.viewer.add_image(data = out, name = f"ves_{dim}D_{sigma}_{cutoff_method}", blending="additive")

    def _merge(self):
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
                    seg = logical_or(seg, image > 0)
                counter += 1
                if counter == 3:
                    break
        self.viewer.add_image(data = seg, name = "merged_segmentation", blending="additive")
        

    def _closing(self):
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

        selected_layer = self.c_closing.currentText()
        for layer in self.viewer.layers:
            if layer.name == selected_layer and type(layer) == Image:
                image = layer.data
                break
        scale = self.s_kernel_size.value()
        out = binary_closing(image, cube(scale))
        self.viewer.add_image(data = out, name = "closed_segmentation", blending="additive")

    def _thinning(self):    # HALVE ONE VALUE
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

        selected_layer = self.c_thinning.currentText()
        for layer in self.viewer.layers:
            if layer.name == selected_layer and type(layer) == Image:
                image = layer.data
                break
        min_thickness = self.s_min_thick.value()/2
        thin = self.s_thin.value()/2
        out = topology_preserving_thinning(image > 0, min_thickness, thin)
        self.viewer.add_image(data = out, name = "thinned_segmentation", blending="additive")

    def _cleaning(self):
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

        selected_layer = self.c_cleaning.currentText()
        for layer in self.viewer.layers:
            if layer.name == selected_layer and type(layer) == Image:
                image = layer.data
                break
        min_size = self.s_min_size.value()/2
        out = remove_small_objects(image > 0, min_size)
        self.viewer.add_image(data = out, name = "cleaned_segmentation", blending="additive")

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
        

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [VesselExpress]