"""
Class View attached with the application
"""
from PySide6.QtWidgets import (QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
                               QLabel, QRadioButton, QCheckBox, QSlider, QPushButton, QGroupBox,
                               QLineEdit, QFileDialog, QMessageBox, QSizePolicy, QButtonGroup,
                               QListWidget, QTextEdit)
from PySide6.QtGui import QTextCursor
from PySide6.QtCore import Qt, Signal

from pathlib import Path
import webbrowser
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from images_alignment.application.callbacks import Callbacks
from images_alignment import REG_MODELS


class View(Callbacks):
    def __init__(self, model):
        super().__init__()
        self.window = QWidget()
        self.model = model

        self.window.setWindowTitle("Image Alignment Application")
        self.window.resize(1400, 800)

        main_layout = QHBoxLayout(self.window)

        self.frame_proc = QFrame(self.window)
        self.frame_proc.setFrameShape(QFrame.StyledPanel)
        self.proc_layout = QVBoxLayout(self.frame_proc)

        self.frame_visu = QFrame()
        self.frame_visu.setFrameShape(QFrame.StyledPanel)
        self.visu_layout = QVBoxLayout(self.frame_visu)

        main_layout.addWidget(self.frame_proc, 1)
        main_layout.addWidget(self.frame_visu, 3)

        # PROCESSING
        ############

        tab_widget = QTabWidget()
        tab_proc = QWidget()
        tab_options = QWidget()
        tab_about = QWidget()
        tab_widget.addTab(tab_proc, 'Processing')
        tab_widget.addTab(tab_options, 'Options')
        tab_options.setFixedHeight(450)
        tab_widget.addTab(tab_about, 'About/Help')
        tab_about.setFixedHeight(420)
        self.proc_layout.addWidget(tab_widget)

        self.init_ui_proc(tab_proc)
        self.init_ui_options(tab_options)
        self.init_ui_about(tab_about)

        # VISUALISATION
        ###############

        self.init_ui_control()

        ratios = [1, 1, 2, 1]
        fig0, ax0 = plt.subplots(1, 4, figsize=(11, 1.5), gridspec_kw={'width_ratios': ratios})
        for i in range(4):
            ax0[i].set_label(i)
            ax0[i].get_xaxis().set_visible(False)
            ax0[i].get_yaxis().set_visible(False)
        self.model.ax = [ax0[i] for i in range(4)]

        self.canvas0 = FigureCanvas(fig0)
        self.canvas0.setFixedHeight(150)
        self.canvas0.draw()
        self.canvas0.mpl_connect('button_press_event', self.select_axis)
        self.canvas0.mpl_connect('scroll_event', self.select_axis)

        self.fig1, self.ax1 = plt.subplots(layout="constrained")
        self.canvas1 = FigureCanvas(self.fig1)
        self.toolbar = NavigationToolbar(self.canvas1, self.window)

        self.canvas1.mpl_connect('button_press_event', self.init_or_remove_rectangle)
        self.canvas1.mpl_connect('motion_notify_event', self.draw_rectangle)
        self.canvas1.mpl_connect('button_release_event', self.set_roi)
        self.canvas1.mpl_connect('button_press_event', self.init_or_remove_line)
        self.canvas1.mpl_connect('motion_notify_event', self.draw_line)
        self.canvas1.mpl_connect('button_press_event', self.add_or_remove_points)
        self.canvas1.mpl_connect('scroll_event', self.zoom)
        self.canvas1.mpl_connect('scroll_event', self.select_axis)

        self.visu_layout.addWidget(self.canvas0)
        self.visu_layout.addWidget(self.canvas1)
        self.visu_layout.addWidget(self.toolbar)

    def init_ui_proc(self, tab_proc):

        self.fselectors = []
        self.rois_entry = [None, None]

        proc_layout = QVBoxLayout(tab_proc)

        for k, label in enumerate(['Fixed image', 'Moving image']):
            frame = QGroupBox(label)
            frame.setStyleSheet("QGroupBox {font-weight: bold}")
            frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            vlayout = QVBoxLayout(frame)

            fselector = FilesSelector(self.window)
            fselector.list_widget.setFixedHeight(50)
            fselector.list_widget.currentRowChanged.connect(fselector.select_item)
            fselector.list_widget.currentRowChanged.connect(lambda _, k=k: self.update_file(k))
            self.fselectors.append(fselector)
            vlayout.addLayout(fselector.layout)

            hlayout = QHBoxLayout()
            label_roi = QLabel("ROI:")
            self.rois_entry[k] = QLineEdit()
            self.rois_entry[k].returnPressed.connect(lambda k=k: self.update_rois(k))
            button_inv = QPushButton('Bin. inversion')
            button_inv.clicked.connect(lambda k=k: self.bin_inversion(k))
            hlayout.addWidget(label_roi)
            hlayout.addWidget(self.rois_entry[k])
            hlayout.addWidget(button_inv)
            vlayout.addLayout(hlayout)

            frame.setLayout(vlayout)
            proc_layout.addWidget(frame)

        frame = QGroupBox('Preprocessing')
        frame.setStyleSheet("QGroupBox {font-weight: bold}")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        vlayout = QVBoxLayout(frame)
        self.group_reg_model = QButtonGroup()
        radios_btn = []
        for i, reg_model in enumerate(REG_MODELS):
            radio_btn = QRadioButton(reg_model)
            radios_btn.append(radio_btn)
            self.group_reg_model.addButton(radio_btn, i)
        radios_btn[0].setChecked(True)
        self.group_reg_model.buttonToggled.connect(self.update_registration_model)
        hlayout = QHBoxLayout()
        [hlayout.addWidget(radio_btn, alignment=Qt.AlignCenter) for radio_btn in radios_btn[:-1]]
        vlayout.addLayout(hlayout)
        vlayout.addWidget(radios_btn[-1], alignment=Qt.AlignCenter)

        check_inv = QCheckBox('INV.')
        check_inv.stateChanged.connect(self.update_inv_reg)
        vlayout.addWidget(check_inv, alignment=Qt.AlignCenter)

        btn_registration = QPushButton('REGISTRATION')
        btn_registration.clicked.connect(self.registration)
        vlayout.addWidget(btn_registration, alignment=Qt.AlignCenter)

        hlayout = QHBoxLayout()
        btn_save = QPushButton('SAVE IMAGES')
        btn_save.clicked.connect(self.save_images)
        btn_save_params = QPushButton('SAVE PARAMS')
        btn_save_params.clicked.connect(self.model.save_params)
        btn_reload_params = QPushButton('RELOAD PARAMS')
        btn_reload_params.clicked.connect(self.reload_params)
        hlayout.addWidget(btn_save)
        hlayout.addWidget(btn_save_params)
        hlayout.addWidget(btn_reload_params)
        vlayout.addLayout(hlayout)

        frame.setLayout(vlayout)
        proc_layout.addWidget(frame)

        frame = QGroupBox('Application')
        frame.setStyleSheet("QGroupBox {font-weight: bold}")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        vlayout = QVBoxLayout(frame)

        btn_select_dir = QPushButton('SELECT DIR. RESULT')
        btn_select_dir.clicked.connect(self.model.set_dirname_res)
        vlayout.addWidget(btn_select_dir, alignment=Qt.AlignCenter)

        hlayout = QHBoxLayout()
        check_fixed = QCheckBox('Fixed registration')
        check_fixed.stateChanged.connect(self.update_fixed_reg)
        btn_apply = QPushButton('APPLY TO ALL')
        btn_apply.clicked.connect(self.apply_to_all)
        hlayout.addWidget(check_fixed)
        hlayout.addWidget(btn_apply)
        vlayout.addLayout(hlayout)

        check_show_results = QCheckBox('Show results')
        check_show_results.stateChanged.connect(self.plot_results)
        vlayout.addWidget(check_show_results, alignment=Qt.AlignCenter)

        frame.setLayout(vlayout)
        proc_layout.addWidget(frame)

        frame = QGroupBox()
        vlayout = QVBoxLayout(frame)
        self.model.terminal = Terminal(frame)
        vlayout.addWidget(self.model.terminal)

        frame.setLayout(vlayout)
        proc_layout.addWidget(frame)

        tab_proc.setLayout(proc_layout)

    def init_ui_options(self, tab_options):

        options_layout = QVBoxLayout(tab_options)

        self.group_angles = []
        self.slider_thresholds = []
        for k, label in enumerate(['Fixed image', 'Moving image']):
            frame = QGroupBox(label)
            frame.setStyleSheet("QGroupBox {font-weight: bold}")
            frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            vlayout = QVBoxLayout(frame)

            hlayout = QHBoxLayout()
            hlayout.addWidget(QLabel('Rotation :'))
            radios_btn = []
            self.group_angles.append(QButtonGroup())
            for i, angle in enumerate([0, 90, 180, 270]):
                radio_btn = QRadioButton(str(angle))
                radios_btn.append(radio_btn)
                self.group_angles[k].addButton(radio_btn, i)
            radios_btn[0].setChecked(True)
            self.group_angles[k].buttonToggled.connect(lambda _, k=k: self.update_angles(k))
            [hlayout.addWidget(radio_btn, alignment=Qt.AlignCenter) for radio_btn in radios_btn]
            vlayout.addLayout(hlayout)

            hlayout = QHBoxLayout()
            hlayout.addWidget(QLabel('Threshold :'))
            self.slider_thresholds.append(QSlider(Qt.Horizontal))
            self.slider_thresholds[k].setValue(50)
            self.slider_thresholds[k].valueChanged.connect(
                lambda val, k=k: self.update_threshold(val, k))
            hlayout.addWidget(self.slider_thresholds[k])
            vlayout.addLayout(hlayout)

            frame.setLayout(vlayout)
            options_layout.addWidget(frame)

        frame = QGroupBox('Resolution')
        frame.setStyleSheet("QGroupBox {font-weight: bold}")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        hlayout = QHBoxLayout(frame)
        hlayout.addWidget(QLabel("Min. image resolution:"))
        self.min_img_res = QLineEdit()
        self.min_img_res.insert(str(self.model.min_img_res))
        self.min_img_res.returnPressed.connect(self.update_resolution)
        hlayout.addWidget(self.min_img_res)

        frame.setLayout(hlayout)
        options_layout.addWidget(frame)

        frame = QGroupBox('Registration')
        frame.setStyleSheet("QGroupBox {font-weight: bold}")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        vlayout = QVBoxLayout(frame)

        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("Max. image size :"))
        self.max_size_reg = QLineEdit()
        self.max_size_reg.insert(str(self.model.max_size_reg))
        self.max_size_reg.returnPressed.connect(self.update_max_size_reg)
        hlayout.addWidget(self.max_size_reg)
        vlayout.addLayout(hlayout)

        hlayout = QHBoxLayout()
        self.tmat_options_cb = []
        for i, key in enumerate(self.model.tmat_options.keys()):
            tmat_option_cb = QCheckBox(key.capitalize())
            tmat_option_cb.setChecked(True)
            if i == 0:
                tmat_option_cb.setEnabled(False)
            else:
                tmat_option_cb.stateChanged.connect(lambda _, i=i: self.update_tmat_options(i))
            self.tmat_options_cb.append(tmat_option_cb)
            hlayout.addWidget(tmat_option_cb)
        vlayout.addLayout(hlayout)

        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("Interpolation :"))
        radios_btn = []
        self.group_interpolations = QButtonGroup()
        for i, interpolation in enumerate(['Default', 'Nearly', 'Linear']):
            radio_btn = QRadioButton(interpolation)
            radios_btn.append(radio_btn)
            self.group_interpolations.addButton(radio_btn, i)
        radios_btn[0].setChecked(True)
        self.group_interpolations.buttonToggled.connect(self.update_interpolation)
        [hlayout.addWidget(radio_btn, alignment=Qt.AlignCenter) for radio_btn in radios_btn]

        vlayout.addLayout(hlayout)

        frame.setLayout(hlayout)
        options_layout.addWidget(frame)

        tab_options.setLayout(options_layout)

    def init_ui_about(self, tab_about):

        about_layout = QVBoxLayout(tab_about)

        text = "This code is dedicated to images alignment.\n\n"

        text += "The sources are accessible in:"
        website_src = r"https://github.com/CEA-MetroCarac/images_alignment"
        website_doc = r"https://cea-metrocarac.github.io/images_alignment/index.html"
        about_layout.addWidget(QLabel(text))
        btn_src = QPushButton(f"{website_src}")
        btn_src.clicked.connect(lambda: webbrowser.open_new(website_src))
        about_layout.addWidget(btn_src, alignment=Qt.AlignCenter)

        text = "The documentation is accessible in:"
        website_doc = r"https://cea-metrocarac.github.io/images_alignment/index.html"
        about_layout.addWidget(QLabel(text))
        btn_doc = QPushButton(f"{website_doc}")
        btn_doc.clicked.connect(lambda: webbrowser.open_new(website_doc))
        about_layout.addWidget(btn_doc, alignment=Qt.AlignCenter)

        label = QLabel("\nShortcuts")
        label.setStyleSheet("QLabel {font-weight: bold}")
        about_layout.addWidget(label, alignment=Qt.AlignCenter)

        shortcuts = {
            "Change thumbnail": ("Scroll Up-Down", ""),
            "Add/Rem. Lines (*)": ("Left/Right Click and Drag", "Juxtaposed images"),
            "Add/Rem. points (*)": ("Left/Right Click", "Fixed/Moving image"),
            "ROI drawing": ("CTRL + Left Click and Drag", "Fixed/Moving image"),
            "ROI deletion": ("CTRL + Right Click", "Fixed/Moving image"),
            "Zooming": ("CTRL + Scroll Up-Down", "ALL images"),
        }

        def draw_separator(k):
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            grid_layout.addWidget(line, k * 2, 0, 1, 3)

        grid_layout = QGridLayout()
        for k, (key, vals) in enumerate(shortcuts.items()):
            draw_separator(k)
            grid_layout.addWidget(QLabel(key), k * 2 + 1, 0)
            grid_layout.addWidget(QLabel(vals[0]), k * 2 + 1, 1)
            grid_layout.addWidget(QLabel(vals[1]), k * 2 + 1, 2)
        draw_separator(k + 1)

        about_layout.addLayout(grid_layout)

        label = QLabel("(*) for 'User-Driven' registration mode only")
        label.setStyleSheet("QLabel {font-style: italic}")
        about_layout.addWidget(label, alignment=Qt.AlignLeft)

        tab_about.setLayout(about_layout)

    def init_ui_control(self):

        widgets_list = []

        self.check_binarized = QCheckBox("Binarized")
        self.check_binarized.setChecked(False)
        self.check_binarized.stateChanged.connect(lambda _: self.update_plots())
        widgets_list.append([self.check_binarized])

        label_resolution = QLabel("Resolution :")
        self.slider_resolution = QSlider(Qt.Horizontal)
        self.slider_resolution.setValue(0)
        self.slider_resolution.valueChanged.connect(self.update_resolution)
        widgets_list.append([label_resolution, self.slider_resolution])

        label_juxtaposition = QLabel("Juxtaposition :")
        self.group_juxtaposition = QButtonGroup()
        radios_btn = []
        for i, juxt_alignment in enumerate(["horizontal", "vertical"]):
            radio_btn = QRadioButton(juxt_alignment)
            radios_btn.append(radio_btn)
            self.group_juxtaposition.addButton(radio_btn, i)
        radios_btn[0].setChecked(True)
        self.group_juxtaposition.buttonToggled.connect(self.update_juxt_alignment)
        widgets_list.append([label_juxtaposition, *radios_btn])

        label_apply_mask = QLabel("Combination :")
        check_apply_mask = QCheckBox("Apply Mask")
        check_apply_mask.setChecked(True)
        check_apply_mask.stateChanged.connect(self.update_apply_mask)
        widgets_list.append([label_apply_mask, check_apply_mask])

        control_layout = QHBoxLayout()
        for wigets in widgets_list:
            frame = QFrame()
            frame.setFrameShape(QFrame.Box)
            frame.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
            hlayout = QHBoxLayout()
            for widget in wigets:
                hlayout.addWidget(widget)
            frame.setLayout(hlayout)
            control_layout.addWidget(frame)
        self.visu_layout.addLayout(control_layout)


class FilesSelector:

    def __init__(self, parent):

        self.parent = parent
        self.fnames = []
        self.list_widget = QListWidget()
        self.layout = QVBoxLayout(self.parent)

        hlayout = QHBoxLayout()
        btn_select = QPushButton("Select Files")
        btn_select.clicked.connect(self.select_files)
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self.remove_selected)
        btn_remove_all = QPushButton("Remove All")
        btn_remove_all.clicked.connect(self.remove_all)
        hlayout.addWidget(btn_select)
        hlayout.addWidget(btn_remove)
        hlayout.addWidget(btn_remove_all)
        self.layout.addLayout(hlayout)
        self.layout.addWidget(self.list_widget)

    def select_files(self, fnames=None):
        if not isinstance(fnames, list):
            fnames, _ = QFileDialog.getOpenFileNames(self.parent, "Select Files")
        if fnames is not None:
            index = len(self.fnames)
            for fname in fnames:
                self.fnames.append(fname)
                self.list_widget.addItem(Path(fname).name)
            self.select_item(index)

    def remove_selected(self):
        selected_rows = [index.row() for index in self.list_widget.selectedIndexes()]
        for row in reversed(selected_rows):
            self.list_widget.takeItem(row)
            self.fnames.pop(row)

    def remove_all(self):
        self.list_widget.clear()
        self.fnames = []

    def select_item(self, index):
        item = self.list_widget.item(index)
        if item:
            item.setSelected(True)
            self.list_widget.setCurrentItem(item)
            self.list_widget.scrollToItem(item)


class Terminal(QTextEdit):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

    def write(self, value):
        self.append(value)


if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication

    fnames = ['dir_1/img_1.png', 'dir_1/img_2.png', 'dir_1/img_3.png',
              'dir_2/img_1.png', 'dir_2/img_2.png']

    qapp = QApplication(sys.argv)
    window = QWidget()
    fselector = FilesSelector(window)
    fselector.select_files(fnames=fnames)
    window.show()
    sys.exit(qapp.exec())
