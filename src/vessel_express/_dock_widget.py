from qtpy.QtWidgets import QComboBox, QLabel, QSizePolicy, QToolBox, QLineEdit, QWidget, QPushButton, QSlider, QHBoxLayout, QVBoxLayout, QScrollArea, QFileDialog, QMessageBox, QCheckBox, QGridLayout, QScrollArea
from inspect import CORO_CLOSED
from qtpy.QtCore import Qt
from napari.layers import Image
from tifffile import imread
import yaml

# packages required by processing functions
from .utils import vesselness_filter
import os
import numpy as np
from glob import glob


class ParameterTuning(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        # Variable to store configuration of last processing run
        self.processing_values = [False, False, False, 0, 0, 0, "", 0, 0, 0, "", 0]

        # Labels
        self.l_title = QLabel("<font color='green'>VesselExpress Segmentation Parameter Tuning:</font>")
        l_scale = QLabel("scale")
        l_sigma_1 = QLabel("sigma")
        l_gamma_1 = QLabel("gamma")
        l_sigma_2 = QLabel("sigma")
        l_gamma_2 = QLabel("gamma")

        # Set tooltips
        """smoothing_layer_tip = (
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
        #TODO: add tooltip for l_10 (voxel)
        
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
        self.l_max_hole_size.setToolTip("the maximum size of holes to be filled")"""

        # Line Edits
        self.le_cutoff_1 = QLineEdit()
        self.le_cutoff_2 = QLineEdit()

        # Sliders
        self.s_scale = QSlider()    # DOUBLED TO MAKE INT WORK
        self.s_scale.setRange(0,50)
        self.s_scale.setValue(0)
        self.s_scale.setOrientation(Qt.Horizontal)
        self.s_scale.setPageStep(4)
        self.s_sigma_1 = QSlider()    # DOUBLED TO MAKE INT WORK
        self.s_sigma_1.setRange(1,20)
        self.s_sigma_1.setValue(1)
        self.s_sigma_1.setOrientation(Qt.Horizontal)
        self.s_sigma_1.setPageStep(2)
        self.s_sigma_2 = QSlider()    # DOUBLED TO MAKE INT WORK
        self.s_sigma_2.setRange(1,20)
        self.s_sigma_2.setValue(1)
        self.s_sigma_2.setOrientation(Qt.Horizontal)
        self.s_sigma_2.setPageStep(2)
        self.s_gamma_1 = QSlider() 
        self.s_gamma_1.setRange(1,1000)
        self.s_gamma_1.setValue(5)
        self.s_gamma_1.setOrientation(Qt.Horizontal)
        self.s_gamma_1.setPageStep(5)
        self.s_gamma_2 = QSlider() 
        self.s_gamma_2.setRange(1,1000)
        self.s_gamma_2.setValue(5)
        self.s_gamma_2.setOrientation(Qt.Horizontal)
        self.s_gamma_2.setPageStep(5)

        # Numeric Labels
        self.n_scale = QLabel()
        self.n_scale.setText("0")
        self.n_scale.setMinimumWidth(35)
        self.n_scale.setMaximumWidth(35)
        self.n_sigma_1 = QLabel()
        self.n_sigma_1.setText("0.5")
        self.n_sigma_1.setMinimumWidth(35)
        self.n_sigma_1.setMaximumWidth(35)
        self.n_sigma_2 = QLabel()
        self.n_sigma_2.setText("0.5")
        self.n_sigma_2.setMinimumWidth(35)
        self.n_sigma_2.setMaximumWidth(35)
        self.n_gamma_1 = QLabel()
        self.n_gamma_1.setText("5")
        self.n_gamma_1.setMinimumWidth(40)
        self.n_gamma_1.setMaximumWidth(40)
        self.n_gamma_2 = QLabel()
        self.n_gamma_2.setText("5")
        self.n_gamma_2.setMinimumWidth(40)
        self.n_gamma_2.setMaximumWidth(40)

        # Link sliders and numeric labels
        self.s_scale.valueChanged.connect(self._update_scale)
        self.s_sigma_1.valueChanged.connect(self._update_sigma_1)
        self.s_sigma_2.valueChanged.connect(self._update_sigma_2)
        self.s_gamma_1.valueChanged.connect(self._update_gamma_1)
        self.s_gamma_2.valueChanged.connect(self._update_gamma_2)

        # Buttons
        btn_load_config = QPushButton("Load config")
        btn_load_config.clicked.connect(self._load_config)
        btn_save_config = QPushButton("Save config")
        btn_save_config.clicked.connect(self._save_config)
        btn_batch_process = QPushButton("Batch process")
        btn_batch_process.clicked.connect(self._batch_process)
        self.btn_show_more = QPushButton("+")
        self.btn_show_more.setMaximumWidth(25)
        self.btn_show_more.clicked.connect(self._toggle_segmentation)
        self.btn_process = QPushButton("Run processing")
        self.btn_process.clicked.connect(self._process)
        self.btn_accept = QPushButton("I accept these values")
        self.btn_accept.clicked.connect(self._lock_processing)
        self.btn_add_widget = QPushButton("Add postprocessing step")
        self.btn_add_widget.clicked.connect(self._add_widget)


        # Horizontal lines
        line_1 = QWidget()
        line_1.setFixedHeight(4)
        line_1.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        line_1.setStyleSheet("background-color: #c0c0c0")
        line_2 = QWidget()
        line_2.setFixedHeight(4)
        line_2.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        line_2.setStyleSheet("background-color: #c0c0c0")
        line_3 = QWidget()
        line_3.setFixedHeight(4)
        line_3.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        line_3.setStyleSheet("background-color: #c0c0c0")

        # Checkboxes
        self.ch_threshold = QCheckBox("threshold")
        self.ch_vesselness_1 = QCheckBox("vesselness 1")
        self.ch_vesselness_2 = QCheckBox("vesselness 2")
        
        # Comboboxes
        cutoff_methods = ["threshold_li","threshold_otsu","threshold_triangle","threshold_constant"]
        self.c_cutoff_1 = QComboBox()
        self.c_cutoff_1.addItems(cutoff_methods)
        self.c_cutoff_1.setMinimumWidth(200)
        self.c_cutoff_1.setMaximumWidth(200)
        def toggle_manual_threshold_1(state):
            if state == "threshold_constant":
                self.le_cutoff_1.show()
            else:
                self.le_cutoff_1.hide()
        self.c_cutoff_1.currentTextChanged.connect(toggle_manual_threshold_1)
        self.c_cutoff_2 = QComboBox()
        self.c_cutoff_2.addItems(cutoff_methods)
        self.c_cutoff_2.setMinimumWidth(200)
        self.c_cutoff_2.setMaximumWidth(200)
        def toggle_manual_threshold_2(state):
            if state == "threshold_constant":
                self.le_cutoff_2.show()
            else:
                self.le_cutoff_2.hide()
        self.c_cutoff_2.currentTextChanged.connect(toggle_manual_threshold_2)
        
        self.segmentation = QWidget()
        self.segmentation.setLayout(QVBoxLayout())
        self.segmentation.layout().addWidget(self.ch_threshold)
        threshold = QWidget()
        threshold.setLayout(QHBoxLayout())
        threshold.layout().addWidget(l_scale)
        threshold.layout().addWidget(self.s_scale)
        threshold.layout().addWidget(self.n_scale)
        self.segmentation.layout().addWidget(threshold)
        self.segmentation.layout().addWidget(self.ch_vesselness_1)
        vess_1_1 = QWidget()
        vess_1_1.setLayout(QHBoxLayout())
        vess_1_1.layout().addWidget(l_sigma_1)
        vess_1_1.layout().addWidget(self.s_sigma_1)
        vess_1_1.layout().addWidget(self.n_sigma_1)
        self.segmentation.layout().addWidget(vess_1_1)
        vess_1_2 = QWidget()
        vess_1_2.setLayout(QHBoxLayout())
        vess_1_2.layout().addWidget(l_gamma_1)
        vess_1_2.layout().addWidget(self.s_gamma_1)
        vess_1_2.layout().addWidget(self.n_gamma_1)
        self.segmentation.layout().addWidget(vess_1_2)
        vess_1_c = QWidget()
        vess_1_c.setLayout(QHBoxLayout())
        vess_1_c.layout().addWidget(self.c_cutoff_1)
        vess_1_c.layout().addWidget(self.le_cutoff_1)
        self.le_cutoff_1.hide()
        self.segmentation.layout().addWidget(vess_1_c)
        self.segmentation.layout().addWidget(self.ch_vesselness_2)
        vess_2_1 = QWidget()
        vess_2_1.setLayout(QHBoxLayout())
        vess_2_1.layout().addWidget(l_sigma_2)
        vess_2_1.layout().addWidget(self.s_sigma_2)
        vess_2_1.layout().addWidget(self.n_sigma_2)
        self.segmentation.layout().addWidget(vess_2_1)
        vess_2_2 = QWidget()
        vess_2_2.setLayout(QHBoxLayout())
        vess_2_2.layout().addWidget(l_gamma_2)
        vess_2_2.layout().addWidget(self.s_gamma_2)
        vess_2_2.layout().addWidget(self.n_gamma_2)
        self.segmentation.layout().addWidget(vess_2_2)
        vess_2_c = QWidget()
        vess_2_c.setLayout(QHBoxLayout())
        vess_2_c.layout().addWidget(self.c_cutoff_2)
        vess_2_c.layout().addWidget(self.le_cutoff_2)
        self.le_cutoff_2.hide()
        self.segmentation.layout().addWidget(vess_2_c)

        # Layouting
        self.content = QWidget()
        self.content.setLayout(QVBoxLayout())
        self.content.layout().addWidget(self.l_title)
        self.content.layout().addWidget(btn_load_config)
        self.content.layout().addWidget(line_1)
        processing = QWidget()
        processing.setLayout(QHBoxLayout())
        processing.layout().addWidget(self.btn_show_more)
        #processing.layout().addWidget(self.btn_process)
        self.content.layout().addWidget(processing)
        self.content.layout().addWidget(line_2)
        self.content.layout().addWidget(self.btn_add_widget)
        self.content.layout().addWidget(line_3)
        batch_save = QWidget()
        batch_save.setLayout(QHBoxLayout())
        batch_save.layout().addWidget(btn_batch_process)
        batch_save.layout().addWidget(btn_save_config)
        self.content.layout().addWidget(batch_save)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.content)
        self.setMinimumSize(400,800)
        self.scroll_area.setWidgetResizable(True)
        
        # Setting layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.scroll_area)
        
        self.viewer.layers.events.inserted.connect(self._update_layer_lists)
        self.viewer.layers.events.removed.connect(self._update_layer_lists)
        self.viewer.layers.events.moved.connect(self._update_layer_lists)

    
    def _toggle_segmentation(self):
        if self.btn_show_more.text() == "+":
            self.btn_show_more.setText("-")
            self.content.layout().itemAt(3).widget().layout().addWidget(self.btn_process)
            self.btn_process.show()
            self.content.layout().insertWidget(4,self.segmentation)
            self.content.layout().insertWidget(5,self.btn_accept)
            self.segmentation.show()
            self.btn_accept.show()
        else:
            self.btn_show_more.setText("+")
            self.content.layout().itemAt(3).widget().layout().removeWidget(self.btn_process)
            self.btn_process.hide()
            self.content.layout().removeWidget(self.segmentation)
            self.content.layout().removeWidget(self.btn_accept)
            self.segmentation.hide()
            self.btn_accept.hide()
        
    def _add_widget(self):
        widget = QWidget()
        combobox_step = QComboBox()
        combobox_step.addItem("closing")
        combobox_step.addItem("hole_removal")
        combobox_step.addItem("thinning")
        combobox_step.addItem("cleaning")
        combobox_step.setMaximumWidth(150)
        combobox_layer = QComboBox()
        for layer in self.viewer.layers:
            combobox_layer.insertItem(0,layer.name)
        combobox_layer.setCurrentIndex(0)
        button = QPushButton("Execute")
        def run():
            self._post_process(widget)
        button.clicked.connect(run)
        
        def show_options():
            if helper_widget.layout().count() > 0:
                old_options = helper_widget.layout().itemAt(0).widget()
                helper_widget.layout().removeWidget(old_options)
                old_options.setVisible(False)
            new_options = QWidget()
            new_options.setLayout(QVBoxLayout())
            
            if combobox_step.currentText() == "closing":
                label = QLabel("kernel:")
                slider = QSlider()
                slider.setRange(1,10)
                slider.setValue(1)
                slider.setOrientation(Qt.Horizontal)
                slider.setPageStep(2)
                numeric_label = QLabel()
                numeric_label.setText("1")
                numeric_label.setMinimumWidth(35)
                numeric_label.setMaximumWidth(35)
                
                def update():
                    numeric_label.setText(str(slider.value()))
                slider.valueChanged.connect(update)
                kernel = QWidget()
                kernel.setLayout(QHBoxLayout())
                kernel.layout().addWidget(slider)
                kernel.layout().addWidget(numeric_label)
                new_options.layout().addWidget(label)
                new_options.layout().addWidget(kernel)

            elif combobox_step.currentText() == "hole_removal":
                label = QLabel("max_size:")
                slider = QSlider()
                slider.setRange(1,100)
                slider.setValue(1)
                slider.setOrientation(Qt.Horizontal)
                slider.setPageStep(2)
                numeric_label = QLabel()
                numeric_label.setText("1")
                numeric_label.setMinimumWidth(35)
                numeric_label.setMaximumWidth(35)
                
                def update():
                    numeric_label.setText(str(slider.value()))
                slider.valueChanged.connect(update)
                max_size = QWidget()
                max_size.setLayout(QHBoxLayout())
                max_size.layout().addWidget(slider)
                max_size.layout().addWidget(numeric_label)
                new_options.layout().addWidget(label)
                new_options.layout().addWidget(max_size)
                
            elif combobox_step.currentText() == "thinning":
                label1 = QLabel("min_thick:")
                slider1 = QSlider()
                slider1.setRange(2,10) # DOUBLED FOR INT REASONS
                slider1.setValue(2)
                slider1.setOrientation(Qt.Horizontal)
                slider1.setPageStep(2)
                numeric_label1 = QLabel()
                numeric_label1.setText("1")
                numeric_label1.setMinimumWidth(35)
                numeric_label1.setMaximumWidth(35)
                
                def update1():
                    numeric_label1.setText(str(slider1.value()/2))
                slider1.valueChanged.connect(update1)
                min_thick = QWidget()
                min_thick.setLayout(QHBoxLayout())
                min_thick.layout().addWidget(slider1)
                min_thick.layout().addWidget(numeric_label1)
                new_options.layout().addWidget(label1)
                new_options.layout().addWidget(min_thick)
                
                label2 = QLabel("thin:")
                slider2 = QSlider()
                slider2.setRange(1,5)
                slider2.setValue(1)
                slider2.setOrientation(Qt.Horizontal)
                slider2.setPageStep(2)
                numeric_label2 = QLabel()
                numeric_label2.setText("1")
                numeric_label2.setMinimumWidth(35)
                numeric_label2.setMaximumWidth(35)
                
                def update2():
                    numeric_label2.setText(str(slider2.value()))
                slider2.valueChanged.connect(update2)
                thin = QWidget()
                thin.setLayout(QHBoxLayout())
                thin.layout().addWidget(slider2)
                thin.layout().addWidget(numeric_label2)
                new_options.layout().addWidget(label2)
                new_options.layout().addWidget(thin)
                
            elif combobox_step.currentText() == "cleaning":
                label = QLabel("min_size:")
                slider = QSlider()
                slider.setRange(1,200)
                slider.setValue(1)
                slider.setOrientation(Qt.Horizontal)
                slider.setPageStep(2)
                numeric_label = QLabel()
                numeric_label.setText("1")
                numeric_label.setMinimumWidth(35)
                numeric_label.setMaximumWidth(35)
                
                def update():
                    numeric_label.setText(str(slider.value()))
                slider.valueChanged.connect(update)
                min_size = QWidget()
                min_size.setLayout(QHBoxLayout())
                min_size.layout().addWidget(slider)
                min_size.layout().addWidget(numeric_label)
                new_options.layout().addWidget(label)
                new_options.layout().addWidget(min_size)
                
            else:
                print("Something went wrong when displaying options!")
            
            helper_widget.layout().insertWidget(0,new_options)

        combobox_step.currentIndexChanged.connect(show_options)
        line = QWidget()
        line.setFixedHeight(4)
        line.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        line.setStyleSheet("background-color: #c0c0c0") 
        widget.setLayout(QVBoxLayout())
        combobox_helper = QWidget()
        combobox_helper.setLayout(QHBoxLayout())
        combobox_helper.layout().addWidget(combobox_step)
        combobox_helper.layout().addWidget(combobox_layer)
        widget.layout().addWidget(combobox_helper)
        helper_widget = QWidget()
        helper_widget.setLayout(QHBoxLayout())
        show_options()
        helper_widget.layout().addWidget(button)
        widget.layout().addWidget(helper_widget)
        widget.layout().addWidget(line)
        self.content.layout().insertWidget(self.content.layout().count() - 1,widget)
        
    def _process(self):
        for layer in self.viewer.layers:
            if type(layer) == Image:
                image = layer.data
                break
        if image is None:
            print("No image layer found!")
            return
        
        # Merge layers from previous run, remove previous merge
        if (self.processing_values[0] or self.processing_values[1] or self.processing_values[2]):
            try:
                self.viewer.layers.remove("merged results")
            except ValueError:
                pass
            if self.processing_values[0]:
                layer1 = self.viewer.layers[self.viewer.layers.index(f"threshold_{self.processing_values[3]}")].data
            else:
                layer1 = None
            if self.processing_values[1]:
                if self.processing_values[6] == "threshold_constant":
                    layer2 = self.viewer.layers[self.viewer.layers.index(f"ves_{self.processing_values[4]}_{self.processing_values[5]}_threshold_constant_{self.processing_values[7]}")].data
                else:
                    layer2 = self.viewer.layers[self.viewer.layers.index(f"ves_{self.processing_values[4]}_{self.processing_values[5]}_{self.processing_values[6]}")].data
            else:
                layer2 = None
            if self.processing_values[2]:
                if self.processing_values[10] == "threshold_constant":
                    layer3 = self.viewer.layers[self.viewer.layers.index(f"ves_{self.processing_values[8]}_{self.processing_values[9]}_threshold_constant_{self.processing_values[11]}")].data
                else:
                    layer3 = self.viewer.layers[self.viewer.layers.index(f"ves_{self.processing_values[8]}_{self.processing_values[9]}_{self.processing_values[10]}")].data
            else:
                layer3 = None
            self._merge(layer1, layer2, layer3)
        
        # Layers only get calculated if the config changed from the last run. Remove old layers if checkmark is removed or data has changed
        if not (not self.processing_values[0] and not self.ch_threshold.isChecked() or self.processing_values[0] and self.ch_threshold.isChecked() and self.s_scale.value()/2 == self.processing_values[3]):
            if self.processing_values[0]:
                self.viewer.layers.remove(f"threshold_{self.processing_values[3]}")
            if self.ch_threshold.isChecked():
                self._threshold(image, self.s_scale.value()/2)
        
        if not (not self.processing_values[1] and not self.ch_vesselness_1.isChecked() or self.processing_values[1] and self.ch_vesselness_1.isChecked() and self.s_sigma_1.value()/2 == self.processing_values[4] and self.s_gamma_1.value() == self.processing_values[5] and self.c_cutoff_1.currentText() == self.processing_values[6]):
            if self.processing_values[1]:
                if self.processing_values[6]() == "threshold_constant":
                    if self.le_cutoff_1.text() == self.processing_values[7]:
                        self.viewer.layers.remove(f"ves_{self.processing_values[4]}_{self.processing_values[5]}_threshold_constant_{self.processing_values[7]}")
                else:
                    self.viewer.layers.remove(f"ves_{self.processing_values[4]}_{self.processing_values[5]}_{self.processing_values[6]}")

            if self.ch_vesselness_1.isChecked():
                self._vesselness(image, self.s_sigma_1.value()/2, self.s_gamma_1.value(), self.c_cutoff_1.currentText(), self.le_cutoff_1.text())
        
        if not (not self.processing_values[2] and not self.ch_vesselness_2.isChecked() or self.processing_values[2] and self.ch_vesselness_2.isChecked() and self.s_sigma_2.value()/2 == self.processing_values[8] and self.s_gamma_2.value() == self.processing_values[9] and self.c_cutoff_2.currentText() == self.processing_values[10]):
            if self.processing_values[2]:
                if self.processing_values[10] == "threshold_constant":
                    if self.le_cutoff_2.text() == self.processing_values[11]:
                        self.viewer.layers.remove(f"ves_{self.processing_values[8]}_{self.processing_values[9]}_threshold_constant_{self.processing_values[11]}")
                else:
                    self.viewer.layers.remove(f"ves_{self.processing_values[8]}_{self.processing_values[9]}_{self.processing_values[10]}")

            if self.ch_vesselness_2.isChecked():
                self._vesselness(image, self.s_sigma_2.value()/2, self.s_gamma_2.value(), self.c_cutoff_2.currentText(), self.le_cutoff_2.text())
        
        # Cache config values
        self.processing_values = [self.ch_threshold.isChecked(), self.ch_vesselness_1.isChecked(), self.ch_vesselness_2.isChecked(), self.s_scale.value()/2,
                                  self.s_sigma_1.value()/2, self.s_gamma_1.value(), self.c_cutoff_1.currentText(), self.le_cutoff_1.text(),
                                  self.s_sigma_2.value()/2, self.s_gamma_2.value(), self.c_cutoff_2.currentText(), self.le_cutoff_2.text()]
        
    def _post_process(self,widget):
        if self.content.layout().count() - 2 > self.content.layout().indexOf(widget):
            widget.setEnabled(False)
        
        layer_name = widget.layout().itemAt(0).widget().layout().itemAt(1).widget().currentText()
        operation = widget.layout().itemAt(0).widget().layout().itemAt(0).widget().currentText()
        if operation == "closing":
            self._closing(self.viewer.layers[self.viewer.layers.index(layer_name)].data, int(widget.layout().itemAt(1).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().layout().itemAt(1).widget().text()))
            pass
        elif operation == "hole_removal":
            self._hole_removal(self.viewer.layers[self.viewer.layers.index(layer_name)].data, int(widget.layout().itemAt(1).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().layout().itemAt(1).widget().text()))
            pass
        elif operation == "thinning":
            self._thinning(self.viewer.layers[self.viewer.layers.index(layer_name)].data, float(widget.layout().itemAt(1).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().layout().itemAt(1).widget().text()), int(widget.layout().itemAt(1).widget().layout().itemAt(0).widget().layout().itemAt(3).widget().layout().itemAt(1).widget().text()))
            pass
        elif operation == "cleaning":
            self._cleaning(self.viewer.layers[self.viewer.layers.index(layer_name)].data, int(widget.layout().itemAt(1).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().layout().itemAt(1).widget().text()))
            pass
        else:
            print("how did you even end up here?")
        self.viewer.layers[self.viewer.layers.index(layer_name)].visible = False
            
            
            
        """print(self.content.layout().count())
        print(self.content.layout().indexOf(widget))"""
        """print(widget.layout().itemAt(0).widget().currentText())
        print(widget.layout().itemAt(1).widget().layout().itemAt(0).widget().text())"""

        
    def _save_config(self):
        filename = QFileDialog.getSaveFileName()
        threshold = {"used" : self.ch_threshold.isChecked(),"scale" : self.s_scale.value()/2}
        vesselness_1 = {"used" : self.ch_vesselness_1.isChecked(),"gamma" : self.s_gamma_1.value(),"sigma" : self.s_sigma_1.value()/2}
        vesselness_2 = {"used" : self.ch_vesselness_2.isChecked(),"gamma" : self.s_gamma_2.value(),"sigma" : self.s_sigma_2.value()/2}
        processing = {"threshold" : threshold,"vesselness 1": vesselness_1,"vesselness 2":vesselness_2}        
        post_processing = []
        for i in range(7,self.content.layout().count()-1):
            if self.content.layout().itemAt(i).widget().layout().itemAt(0).widget().layout().itemAt(0).widget().currentText() == "thinning":
                post_processing.append({"thinning" : { "min_thick" : self.content.layout().itemAt(i).widget().layout().itemAt(1).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().layout().itemAt(1).widget().text(),"thin" : self.content.layout().itemAt(i).widget().layout().itemAt(1).widget().layout().itemAt(0).widget().layout().itemAt(3).widget().layout().itemAt(1).widget().text()}})
            else:
                post_processing.append({self.content.layout().itemAt(i).widget().layout().itemAt(0).widget().layout().itemAt(0).widget().currentText() : {self.content.layout().itemAt(i).widget().layout().itemAt(1).widget().layout().itemAt(0).widget().layout().itemAt(0).widget().text() : self.content.layout().itemAt(i).widget().layout().itemAt(1).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().layout().itemAt(1).widget().text()}})
        config = [{"post-processing" : post_processing,"processing" : processing}]
        
        with open(filename[0], 'w') as file:
            yaml.dump(config,file)
        pass
    
    def _load_config(self):
        filename = QFileDialog.getOpenFileName()
        try:
            with open(filename[0]) as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            print("File not found, try again")
            return
        self.ch_threshold.setChecked(config["processing"]["threshold"]["used"])
        self.s_scale.setValue(int(config["processing"]["threshold"]["scale"]*2))
        self.ch_vesselness_1.setChecked(config["processing"]["vesselness 1"]["used"])
        self.s_gamma_1.setValue(config["processing"]["vesselness 1"]["gamma"])
        self.s_sigma_1.setValue(int(config["processing"]["vesselness 1"]["sigma"]*2))
        self.ch_vesselness_2.setChecked(config["processing"]["vesselness 2"]["used"])
        self.s_gamma_2.setValue(config["processing"]["vesselness 2"]["gamma"])
        self.s_sigma_2.setValue(int(config["processing"]["vesselness 2"]["sigma"]*2))
        if not isinstance(config["post-processing"],list):
            post_processing = list(config["post-processing"])
        else:
            post_processing = config["post-processing"]
        for i in range(len(post_processing)):
            self._add_widget()
            j = i + 7
            if self.btn_show_more.text() == "-":
                j += 2
            if str(list(post_processing[i].keys())[0]) == "closing":
                self.content.layout().itemAt(j).widget().layout().itemAt(0).widget().layout().itemAt(0).widget().setCurrentText("closing")
                self.content.layout().itemAt(j).widget().layout().itemAt(1).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().layout().itemAt(0).widget().setValue(post_processing[i]["closing"]["kernel"])
                pass
            elif str(list(post_processing[i].keys())[0]) == "hole_removal":
                self.content.layout().itemAt(j).widget().layout().itemAt(0).widget().layout().itemAt(0).widget().setCurrentText("hole_removal")
                self.content.layout().itemAt(j).widget().layout().itemAt(1).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().layout().itemAt(0).widget().setValue(post_processing[i]["hole_removal"]["max_size"])
                pass
            elif str(list(post_processing[i].keys())[0]) == "thinning":
                self.content.layout().itemAt(j).widget().layout().itemAt(0).widget().layout().itemAt(0).widget().setCurrentText("thinning")
                self.content.layout().itemAt(j).widget().layout().itemAt(1).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().layout().itemAt(0).widget().setValue(int(post_processing[i]["thinning"]["min_thick"]*2))
                self.content.layout().itemAt(j).widget().layout().itemAt(1).widget().layout().itemAt(0).widget().layout().itemAt(3).widget().layout().itemAt(0).widget().setValue(post_processing[i]["thinning"]["thin"])
                pass
            elif str(list(post_processing[i].keys())[0]) == "cleaning":
                self.content.layout().itemAt(j).widget().layout().itemAt(0).widget().layout().itemAt(0).widget().setCurrentText("cleaning")
                self.content.layout().itemAt(j).widget().layout().itemAt(1).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().layout().itemAt(0).widget().setValue(post_processing[i]["cleaning"]["min_size"])
                pass
            else:
                print("never should have come here")
            
    
    def _batch_process(self):
        pass
        
    def _update_layer_lists(self,event):
        j = 7
        if self.btn_show_more.text() == "-":
            j += 2
        for i in range(j,self.content.layout().count()-1):
            if event.type == "inserted":
                self.content.layout().itemAt(i).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().insertItem(0,event.value.name)
                self.content.layout().itemAt(i).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().setCurrentIndex(0)

            elif event.type == "removed":
                self.content.layout().itemAt(i).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().removeItem(len(self.viewer.layers) - event.index)

            elif event.type == "moved":
                selection =  self.content.layout().itemAt(i).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().currentIndex()
                self.content.layout().itemAt(i).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().removeItem(len(self.viewer.layers) - 1 - event.index)
                self.content.layout().itemAt(i).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().insertItem(len(self.viewer.layers) - 1 - event.new_index,event.value.name)
                if event.index == selection:
                    self.content.layout().itemAt(i).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().setCurrentIndex(event.new_index)
                elif event.new_index == selection:
                    self.content.layout().itemAt(i).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().setCurrentIndex(event.index)
            else:
                print("Something went wrong when identifying the type of event emitted!")
                
    def _lock_processing(self):
        self.segmentation.setEnabled(False)
        self.btn_process.setEnabled(False)
        
        # Merge layers from previous run, remove previous merge
        if (self.processing_values[0] or self.processing_values[1] or self.processing_values[2]):
            try:
                self.viewer.layers.remove("merged results")
            except ValueError:
                pass
            if self.processing_values[0]:
                layer1 = self.viewer.layers[self.viewer.layers.index(f"threshold_{self.processing_values[3]}")].data
            else:
                layer1 = None
            if self.processing_values[1]:
                if self.processing_values[6] == "threshold_constant":
                    layer2 = self.viewer.layers[self.viewer.layers.index(f"ves_{self.processing_values[4]}_{self.processing_values[5]}_threshold_constant_{self.processing_values[7]}")].data
                else:
                    layer2 = self.viewer.layers[self.viewer.layers.index(f"ves_{self.processing_values[4]}_{self.processing_values[5]}_{self.processing_values[6]}")].data
            else:
                layer2 = None
            if self.processing_values[2]:
                if self.processing_values[10] == "threshold_constant":
                    layer3 = self.viewer.layers[self.viewer.layers.index(f"ves_{self.processing_values[8]}_{self.processing_values[9]}_threshold_constant_{self.processing_values[11]}")].data
                else:
                    layer3 = self.viewer.layers[self.viewer.layers.index(f"ves_{self.processing_values[8]}_{self.processing_values[9]}_{self.processing_values[10]}")].data
            else:
                layer3 = None
            self._merge(layer1, layer2, layer3)
        
        if self.processing_values[0]:
            self.viewer.layers.remove(f"threshold_{self.processing_values[3]}")
        if self.processing_values[1]:
            if self.processing_values[6] == "threshold_constant":
                self.viewer.layers.remove(f"ves_{self.processing_values[4]}_{self.processing_values[5]}_threshold_constant_{self.processing_values[7]}")
            else:
                self.viewer.layers.remove(f"ves_{self.processing_values[4]}_{self.processing_values[5]}_{self.processing_values[6]}")
        if self.processing_values[2]:
            if self.processing_values[10] == "threshold_constant":
                self.viewer.layers.remove(f"ves_{self.processing_values[8]}_{self.processing_values[9]}_threshold_constant_{self.processing_values[11]}")
            else:
                self.viewer.layers.remove(f"ves_{self.processing_values[8]}_{self.processing_values[9]}_{self.processing_values[10]}")
            

    # Slider update functions
    def _update_scale(self):
        self.n_scale.setText(str(self.s_scale.value()/2))

    def _update_sigma_1(self):
        self.n_sigma_1.setText(str(self.s_sigma_1.value()/2))
        
    def _update_sigma_2(self):
        self.n_sigma_2.setText(str(self.s_sigma_2.value()/2))

    def _update_gamma_1(self):
        self.n_gamma_1.setText(str(self.s_gamma_1.value()))
        
    def _update_gamma_2(self):
        self.n_gamma_2.setText(str(self.s_gamma_2.value()))

    # Button onclick functions
    def _smoothing(self, preset = False, data = ""):
        """
        perform edge preserving smoothing
        """
        
        from aicssegmentation.core.pre_processing_utils import  edge_preserving_smoothing_3d

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

    def _isotropic(self):
        from skimage.transform import rescale
        selected_layer = self.c_isotropic.currentText()
        for layer in self.viewer.layers:
            if layer.name == selected_layer and type(layer) == Image:
                image = layer.data
                break
        x = float(self.li_x.displayText())
        y = float(self.li_y.displayText())
        z = float(self.li_z.displayText())
        largest_dim = max(1/x, 1/y, 1/z)
        out = rescale(image, scale=(z * largest_dim, y * largest_dim, x * largest_dim), order=1)
        self.viewer.add_image(data = out, name = f"isotropic_{x}_{y}_{z}", blending="additive")

    def _threshold(self, image, scale):   # HALVE VALUE
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
        None
        """

        thresh = image.mean() + scale * image.std()
        out = image > thresh
        out = 1 * out       # convert from bool to int
        self.viewer.add_image(data = out, name = f"threshold_{scale}", blending="additive")

    def _vesselness(self, image, sigma, gamma, cutoff_method, cutoff_value):  # HALVE VALUE
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
        None
        """

        if cutoff_method == "threshold_constant":
            self.viewer.add_image(data = out, name = f"ves_{sigma}_{gamma}_{cutoff_method}_{cutoff_value}", blending="additive")
            pass
        else:
            out = vesselness_filter(image, 3, sigma, gamma, cutoff_method)
            out = 1 * out
            self.viewer.add_image(data = out, name = f"ves_{sigma}_{gamma}_{cutoff_method}", blending="additive")
        """out = vesselness_filter(image, 3, sigma, gamma, cutoff_method)
        out = 1 * out
        self.viewer.add_image(data = out, name = f"ves_{sigma}_{gamma}_{cutoff_method}", blending="additive")"""

    def _merge(self, layer1, layer2, layer3):
        if not layer1 is None:
            if layer2 is None:
                if layer3 is None:
                    print("merged single layer, whatever that means")
                    self.viewer.add_image(data = layer1 > 0, name = "merged results", blending="additive")
                    return
                self.viewer.add_image(data = np.logical_or(layer1 > 0, layer3 > 0), name = "merged results", blending="additive")
                return
            if layer3 is None:
                self.viewer.add_image(data = np.logical_or(layer1 > 0, layer2 > 0), name = "merged results", blending="additive")
                return
            self.viewer.add_image(data = np.logical_or(np.logical_or(layer1 > 0, layer2 > 0), layer3 > 0), name = "merged results", blending="additive")
        elif not layer2 is None:
            if layer3 is None:
                print("merged single layer, whatever that means")
                self.viewer.add_image(data = layer2 > 0, name = "merged results", blending="additive")
                return 
            self.viewer.add_image(data = np.logical_or(layer2 > 0, layer3 > 0), name = "merged results", blending="additive")
        elif not layer3 is None:
            print("merged single layer, whatever that means")
            self.viewer.add_image(data = layer3 > 0, name = "merged results", blending="additive")
        else:
            print("can't merge no layers")
        

    def _closing(self, image, kernel):
        """
        perform morphological closing to remove small gaps in segmentation
        Parameters:
        -------------
        image: np.ndarray
            the image to be applied on
        kernel: int
            the kernel size of the closing operation
        Return
        -------------
        None
        """
        
        from skimage.morphology import binary_closing, cube
        out = binary_closing(image, cube(kernel))
        self.viewer.add_image(data = out, name = f"closing_{kernel}", blending="additive")
    
    def _hole_removal(self, image, max_size):
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
        None
        """
        from aicssegmentation.core.utils import hole_filling
        out = hole_filling(image, hole_min=1, hole_max=max_size, fill_2d=True)
        self.viewer.add_image(data = out, name = f"hole_filled_{max_size}", blending="additive")

    def _thinning(self, image, min_thickness, thin):
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
        None
        """
        
        from aicssegmentation.core.utils import topology_preserving_thinning
        out = topology_preserving_thinning(image > 0, min_thickness, thin)
        self.viewer.add_image(data = out, name = f"thinned_{min_thickness}_{thin}", blending="additive")

    def _cleaning(self, image, min_size):
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
        None
        """

        from skimage.morphology import remove_small_objects
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
        self.l_directory = QLabel("No file displayed yet")


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

        line_1 = QWidget()
        line_1.setFixedHeight(4)
        line_1.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        line_1.setStyleSheet("background-color: #c0c0c0")

        line_2 = QWidget()
        line_2.setFixedHeight(4)
        line_2.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        line_2.setStyleSheet("background-color: #c0c0c0")
        
        # Layouting
        self.content = QWidget()
        self.content.setLayout(QVBoxLayout())
        self.content.layout().addWidget(self.l_title)
        self.content.layout().addWidget(self.btn_dir)
        self.content.layout().addWidget(line_1)
        self.content.layout().addWidget(self.l_directory)
        self.content.layout().addWidget(self.btn_next)
        self.content.layout().addWidget(self.l_evaluate)
        self.content.layout().addWidget(self.c_eval)
        self.content.layout().addWidget(line_2)
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
        