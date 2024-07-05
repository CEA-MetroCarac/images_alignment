"""
Application for images registration
"""
import os
from copy import deepcopy
from pathlib import Path
import json
from tkinter import filedialog
import panel as pn
import numpy as np
import imageio.v3 as iio
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from pystackreg import StackReg
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import warp, AffineTransform
from skimage.feature import SIFT, match_descriptors, plot_matches
from skimage.measure import ransac
from scipy.interpolate import RegularGridInterpolator

AXES_TITLES = ['Fixed image', 'Moving image', 'Combined image', None]
FLOW_MODES = ['Flow Auto', 'Iterative']
VIEW_MODES = [['Gray', 'Binarized', 'Matching (SIFT)'], AXES_TITLES[:3]]
REG_MODELS = ['StackReg', 'SIFT']
# REG_MODE = ["FIXED REG.", "VARIABLE REG."]
STREG = StackReg(StackReg.AFFINE)
STEP = 1  # translation increment
ANGLE = np.deg2rad(1)  # rotation angular increment
COEF1, COEF2 = 0.99, 1.01  # scaling coefficient

KEYS = ['fnames_fixed', 'fnames_moving',
        'thresholds', 'bin_inversions', 'mode_auto', 'tmat']

pn.extension('terminal', inline=True)
pn.pane.Str.align = 'center'
pn.pane.Markdown.align = 'center'
pn.pane.Matplotlib.align = 'center'
pn.widgets.Button.align = 'center'
pn.widgets.RadioButtonGroup.align = 'center'
pn.widgets.FloatSlider.align = 'center'
pn.widgets.Checkbox.align = 'center'
pn.widgets.TextInput.align = 'center'
pn.widgets.FileInput.align = 'center'
pn.Row.align = 'center'
pn.Column.align = 'center'


def gray_conversion(img):
    """ Convert RGBA or RGB image to gray image """
    if img.ndim == 4:
        img = rgba2rgb(img)
    if img.ndim == 3:
        img = rgb2gray(img)
    return img


def image_normalization(img):
    """ Normalize image in range [0., 1.] """
    vmin, vmax = img.min(), img.max()
    return (img - vmin) / (vmax - vmin)


def edges_trend(img):
    """ Estimate if the average value along the image edges is > 0.5 """
    shape = img.shape
    mask = np.ones_like(img, dtype=bool)
    mask[1:-1, 1:-1] = 0
    sum_edges = np.sum(img[mask])
    return (sum_edges / (2 * (shape[0] + shape[1] - 2))) > 0.5


def padding(img1, img2):
    """ Add image padding """
    shape1 = img1.shape
    shape2 = img2.shape

    hmax = max(img1.shape[0], img2.shape[0])
    wmax = max(img1.shape[1], img2.shape[1])

    img1_pad = np.pad(img1, ((0, hmax - shape1[0]), (0, wmax - shape1[1])))
    img2_pad = np.pad(img2, ((0, hmax - shape2[0]), (0, wmax - shape2[1])))

    return img1_pad, img2_pad


def interpolation(img1, img2):
    """
    Interpolate the low resolution image on the high resolution image support

    Parameters
    ----------
    img1, img2: numpy.ndarray((m, n)), numpy.ndarray((p, q))

    Returns
    -------
    img1_int, img2_int: numpy.ndarrays((r, s))
        The linearly interpolated arrays on the high resolution support of size
        (r, s) = max((m, n), (p, q))
    success: bool
        Status related to differentiation between low and high resolution images
    """
    shape1 = np.array(img1.shape)
    shape2 = np.array(img2.shape)
    success = True

    if (shape1 == shape2).all():
        return img1, img2, success
    elif (shape1 <= shape2).all():
        img_lr = img1
        img_hr = img2
    elif (shape1 >= shape2).all():
        img_lr = img2
        img_hr = img1
    else:
        success = False
        return None, None, success

    # image padding to have same scale ratio between images
    ratio0 = img_hr.shape[0] / img_lr.shape[0]
    ratio1 = img_hr.shape[1] / img_lr.shape[1]
    if ratio0 < ratio1:
        pad = int((img_hr.shape[1] / ratio0) - img_lr.shape[1])
        img_lr = np.pad(img_lr, ((0, 0), (int(pad / 2), pad - int(pad / 2))))
    else:
        pad = int((img_hr.shape[0] / ratio1) - img_lr.shape[0])
        img_lr = np.pad(img_lr, ((int(pad / 2), pad - int(pad / 2)), (0, 0)))

    # low resolution support
    row_lr = np.linspace(0, 1, img_lr.shape[0])
    col_lr = np.linspace(0, 1, img_lr.shape[1])

    # high resolution support
    row_hr = np.linspace(0, 1, img_hr.shape[0])
    col_hr = np.linspace(0, 1, img_hr.shape[1])

    # interpolation
    interp = RegularGridInterpolator((row_lr, col_lr), img_lr)
    rows_hr, cols_hr = np.meshgrid(row_hr, col_hr, indexing='ij')
    pts = np.vstack((rows_hr.ravel(), cols_hr.ravel())).T
    img_lr_int = interp(pts).reshape(img_hr.shape)

    if (shape1 <= shape2).all():
        return img_lr_int, img_hr, success
    else:
        return img_hr, img_lr_int, success


def sift(img1, img2, model_class=None):
    """
    SIFT feature detection and descriptor extraction

    Parameters
    ----------
    img1, img2: numpy.ndarray((m, n)), numpy.ndarray((p, q))
        The input images
    model_class: Objet
        'model_class' used by `RANSAC
        <https://scikit-image.org/docs/stable/api/skimage.measure.html>`_.
        If None, consider ``AffineTransform`` from skimage.transform.

    Returns
    -------
    tmat: numpy.ndarrays((3, 3))
        The related transformation matrix
    keypoints: list of 2 numpy.ndarray((n, 2)
        Keypoints coordinates as (row, col) related to the 2 input images.
    descriptors: list of 2 numpy.ndarray((n, p)
        Descriptors associated with the keypoints.
    matches: numpy.ndarray((q, 2))
        Indices of corresponding matches returned by
        skimage.feature.match_descriptors.

    """
    if model_class is None:
        model_class = AffineTransform

    sift_ = SIFT(upsampling=2)

    keypoints = []
    descriptors = []
    for img in [img1, img2]:
        sift_.detect_and_extract(img)
        keypoints.append(sift_.keypoints)
        descriptors.append(sift_.descriptors)

    matches = match_descriptors(descriptors[0], descriptors[1],
                                cross_check=True, max_ratio=0.8)

    src = keypoints[0][matches[:, 0]][:, ::-1]
    dst = keypoints[1][matches[:, 1]][:, ::-1]
    tmat = ransac((src, dst), model_class,
                  min_samples=4, residual_threshold=2)[0].params

    return tmat, keypoints, descriptors, matches


class App:
    """
    Application dedicated to images alignment

    Parameters
    ----------
    fnames: iterable of 2 str, optional
        Images pathnames to handle
    thresholds: iterable of 2 floats, optional
        Thresholds used to binarize the images
    bin_inversions: iterable of 2 bools, optional
        Activation keywords to reverse the image binarization
    mode_auto: bool, optional
        Activation keyword to realize image processing in automatic mode
    """

    def __init__(self, fnames_fixed=None, fnames_moving=None,
                 thresholds=None, bin_inversions=None, mode_auto=True):

        self.fnames_tot = [fnames_fixed, fnames_moving]
        self.fnames = [None, None]
        if self.fnames_tot[0] is not None:
            self.fnames[0] = self.fnames_tot[0][0]
        if self.fnames_tot[1] is not None:
            self.fnames[1] = self.fnames_tot[1][0]

        self.thresholds = thresholds or [0.5, 0.5]
        self.bin_inversions = bin_inversions or [False, False]
        self.mode_auto = mode_auto

        self.window = None
        self.ax = [None, None, None, None]
        self.view_modes = ['Gray', 'Fixed image']
        self.imgs = [None, None]
        self.cropping_areas = [None, None]
        self.imgs_bin = [None, None]
        self.registration_model = 'StackReg'
        self.keypoints = None
        self.descriptors = None
        self.matches = None
        self.tmat = np.identity(3)
        self.img_reg = None
        self.results = {}
        self.dir_results = None
        self.fixed_reg = False
        self.input_dirpath = None
        self.output_dirpath = None
        self.input_fnames = None
        self.save_fname = None
        self.reload_fname = None

        self.mpl_panes = [None, None, None, None]
        self.result_str = pn.pane.Str(None, align='center',
                                      styles={'text-align': 'center'})
        self.terminal = pn.widgets.Terminal(align='center',
                                            height=100,
                                            sizing_mode='stretch_width',
                                            options={"theme": {
                                                'background': '#F3F3F3',
                                                'foreground': '#000000'}})

        figs = [Figure(figsize=(4, 4)),
                Figure(figsize=(4, 4)),
                Figure(figsize=(4, 4)),
                Figure(figsize=(12, 12))]

        for k in range(4):
            self.ax[k] = figs[k].subplots()
            self.ax[k].set_title(AXES_TITLES[k])
            self.ax[k].autoscale(tight=True)
            self.mpl_panes[k] = pn.pane.Matplotlib(figs[k], dpi=80, tight=True,
                                                   align='center',
                                                   sizing_mode='stretch_both')

        range_slider = pn.widgets.RangeSlider(width=150, step=0.01,
                                              show_value=False,
                                              bar_color='#0072b5')

        file_inputs = []
        reinit_buttons = []
        self.h_range_sliders = []
        self.v_range_sliders = []
        crop_buttons = []
        thresh_sliders = []
        reverse_checks = []
        for k in range(2):
            file_input = pn.widgets.FileInput(accept='image/*')
            file_input.param.watch(
                lambda event, k=k: self.update_file(k, event.new), 'value')
            file_input.value = self.fnames_tot[k]
            file_inputs.append(file_input)

            reinit_button = pn.widgets.Button(name='REINIT')
            reinit_button.on_click(lambda event, k=k: self.reinit(k))
            reinit_buttons.append(reinit_button)

            self.h_range_sliders.append(deepcopy(range_slider))
            self.v_range_sliders.append(deepcopy(range_slider))
            self.h_range_sliders[k].param.watch(
                lambda event, k=k: self.cropping(k, show_only=True), 'value')
            self.v_range_sliders[k].param.watch(
                lambda event, k=k: self.cropping(k, show_only=True), 'value')

            crop_button = pn.widgets.Button(name='CROP')
            crop_button.on_click(lambda event, k=k: self.cropping(k))
            crop_buttons.append(crop_button)

            thresh_slider = pn.widgets.FloatSlider(name='threshold ', step=0.01,
                                                   value=self.thresholds[k],
                                                   bar_color='#0072b5',
                                                   width=150)
            thresh_slider.param.watch(
                lambda event, k=k: self.update_threshold(k, event), 'value')
            thresh_sliders.append(thresh_slider)

            reverse_check = pn.widgets.Checkbox(name='Reversed',
                                                value=self.bin_inversions[k],
                                                height=0)
            reverse_check.param.watch(
                lambda event, k=k: self.update_reverse(k, event), 'value')
            reverse_checks.append(reverse_check)

        value = FLOW_MODES[not self.mode_auto]
        mode_auto_check = pn.widgets.RadioButtonGroup(options=FLOW_MODES,
                                                      button_style='outline',
                                                      button_type='primary',
                                                      value=value)
        mode_auto_check.param.watch(self.update_mode_auto, 'value')

        self.resizing_button = pn.widgets.Button(name='RESIZING', margin=2)
        self.resizing_button.on_click(lambda event: self.resizing())

        self.binarize_button = pn.widgets.Button(name='BINARIZATION', margin=2)
        self.binarize_button.on_click(lambda event: self.binarization())

        self.register_button = pn.widgets.Button(name='REGISTRATION', margin=2)
        self.register_button.on_click(lambda event: self.registration())

        value = self.registration_model
        self.reg_models = pn.widgets.RadioButtonGroup(options=REG_MODELS,
                                                      button_style='outline',
                                                      button_type='primary',
                                                      value=value)
        self.reg_models.param.watch(self.update_registration_model, 'value')

        transl_up_but = pn.widgets.Button(name='▲')
        transl_down_but = pn.widgets.Button(name='▼')
        transl_left_but = pn.widgets.Button(name='◄', margin=5)
        transl_right_but = pn.widgets.Button(name='►', margin=5)
        transl_up_but.on_click(lambda _, mode='up': self.translate(mode))
        transl_down_but.on_click(lambda _, mode='down': self.translate(mode))
        transl_left_but.on_click(lambda _, mode='left': self.translate(mode))
        transl_right_but.on_click(lambda _, mode='right': self.translate(mode))

        rot_clock_button = pn.widgets.Button(name='↻')
        rot_anticlock_button = pn.widgets.Button(name='↺')
        rot_clock_button.on_click(lambda event: self.rotate(ANGLE))
        rot_anticlock_button.on_click(lambda event: self.rotate(-ANGLE))

        self.xc_rel = pn.widgets.FloatInput(value=0.5, width=60)
        self.yc_rel = pn.widgets.FloatInput(value=0.5, width=60)

        view_mode = pn.widgets.RadioButtonGroup(options=VIEW_MODES[0],
                                                button_style='outline',
                                                button_type='primary',
                                                value=self.view_modes[0],
                                                align='center')
        view_mode.param.watch(self.update_view_mode, 'value')

        view_mode_zoom = pn.widgets.RadioButtonGroup(options=VIEW_MODES[1],
                                                     button_style='outline',
                                                     button_type='primary',
                                                     value=self.view_modes[1],
                                                     align='center')
        view_mode_zoom.param.watch(self.update_view_mode_zoom, 'value')

        select_dir_button = pn.widgets.Button(name='SELECT DIR. RESULT')
        select_dir_button.on_click(lambda event: self.select_dir_result())

        apply_button = pn.widgets.Button(name='APPLY & SAVE')
        apply_button.on_click(lambda event: self.apply())

        fixed_reg_check = pn.widgets.Checkbox(name='Fixed registration',
                                              value=self.fixed_reg)
        # reverse_check.param.watch(
        #     lambda event, k=k: self.update_reverse(k, event), 'value')
        # reverse_checks.append(reverse_check)

        save_button = pn.widgets.Button(name='SAVE MODEL')
        save_button.on_click(lambda event: self.save())
        self.save_input = pn.widgets.FileInput(accept='.json')
        self.save_input.param.watch(self.update_save_fname, 'value')

        reload_button = pn.widgets.Button(name='RELOAD MODEL')
        reload_button.on_click(
            lambda _, fname=self.reload_fname: self.reload(fname=fname))
        reload_input = pn.widgets.FileInput(accept='.json')
        reload_input.param.watch(self.update_reload_fname, 'value')

        boxes = []

        text = ['FIXED IMAGE', 'MOVING IMAGE']
        h_label = pn.pane.Str("H:")
        v_label = pn.pane.Str("V:")
        for k in range(2):
            img_box_title = pn.pane.Markdown(f"**{text[k]}**")
            img_crop = pn.Row(pn.Column(h_label, v_label),
                              pn.Column(self.h_range_sliders[k],
                                        self.v_range_sliders[k]),
                              crop_buttons[k])
            img_param = pn.Row(thresh_sliders[k], reverse_checks[k])
            img_box = pn.WidgetBox(img_box_title,
                                   file_inputs[k], reinit_buttons[k],
                                   img_crop, img_param,
                                   margin=(5, 0), width=350)
            boxes.append(img_box)

        proc_box_title = pn.pane.Markdown("**IMAGES PROCESSING**")

        transl_box = pn.GridBox(*[None for _ in range(3)], nrows=3, width=100)
        transl_box[0] = transl_up_but
        transl_box[1] = pn.Row(transl_left_but, transl_right_but)
        transl_box[2] = transl_down_but

        rot_box = pn.GridBox(*[None for _ in range(6)], nrows=3, ncols=2)
        rot_box[0] = rot_clock_button
        rot_box[1] = rot_anticlock_button
        rot_box[2] = pn.pane.Str("xc:")
        rot_box[3] = self.xc_rel
        rot_box[4] = pn.pane.Str("yc:")
        rot_box[5] = self.yc_rel

        proc_box = pn.Column(pn.Row(mode_auto_check, self.reg_models),
                             pn.Row(self.resizing_button,
                                    self.binarize_button,
                                    self.register_button),
                             pn.Row(transl_box, pn.Spacer(width=30), rot_box),
                             self.result_str)
        proc_box = pn.WidgetBox(proc_box_title, proc_box, margin=(5, 0),
                                width=350)
        boxes.append(proc_box)

        appl_box_title = pn.pane.Markdown("**APPLICATION**")

        appl_box = pn.WidgetBox(appl_box_title,
                                # row_input_dirpath,
                                # row_output_dirpath,
                                # select_dir_button,
                                pn.Row(apply_button, fixed_reg_check),
                                pn.Row(save_button, reload_button),
                                # self.terminal,
                                margin=(5, 20), width=350)

        col1 = pn.Column(*boxes, appl_box, width=350)

        col2 = pn.Column(view_mode,
                         self.mpl_panes[0],
                         self.mpl_panes[1],
                         self.mpl_panes[2], width=350, align='center')

        col3 = pn.Column(view_mode_zoom,
                         self.mpl_panes[3],
                         self.terminal,
                         align='center',
                         sizing_mode='stretch_width')

        self.window = pn.Row(col1, col2, col3, sizing_mode='stretch_both')
        self.update_disabled()

    def update_view_mode(self, event):
        """ Update the 'view_mode' attribute and replot """
        self.view_modes[0] = event.new
        [self.update_plot(i) for i in range(3)]
        self.update_plot_zoom()

    def update_view_mode_zoom(self, event):
        """ Update the 'view_mode' attribute and replot """
        self.view_modes[1] = event.new
        self.update_plot_zoom()

    def update_mode_auto(self, event):
        """ Update the 'mode_auto' attribute """
        self.mode_auto = event.new == FLOW_MODES[0]
        self.update_disabled()

    def update_disabled(self):
        """ Change the disabled status of some buttons """
        self.resizing_button.disabled = self.mode_auto
        self.binarize_button.disabled = self.mode_auto
        self.register_button.disabled = self.mode_auto

    def update_file(self, k, fnames):
        """ Load the k-th image file """
        if not isinstance(fnames, list):
            fnames = [fnames]
        try:
            img = iio.imread(fnames[0])
            self.imgs[k] = self.imgs_bin[k] = None
            self.tmat = self.keypoints = self.descriptors = self.matches = None
            self.img_reg = None
            self.results = {}
        except Exception as _:
            self.terminal.write(f"Failed to load {fnames[0]}\n\n")
            return

        self.fnames_tot[k] = fnames
        self.fnames[k] = fnames[0]

        # image normalization in range [0, 1]
        self.imgs[k] = image_normalization(gray_conversion(img))

        self.update_plot(k)
        self.update_plot(2)

        if self.mode_auto:
            self.resizing()

    def update_plot(self, k, patch=None):
        """ Update the k-th ax """

        self.ax[k].clear()

        img = None
        mode = self.view_modes[0]
        title = AXES_TITLES[k]

        if k in [0, 1]:
            img = self.imgs[k]
            title += f" - {self.fnames[k].name}"

            if mode == 'Binarized':
                img_bin = self.imgs_bin[k]
                if img_bin is None:
                    img_bin = self.binarization_k(k)
                shape = img_bin.shape
                img = np.zeros((shape[0], shape[1], 3))
                RGB_channel = [1, 0][k]  # k=0 -> Green, k=1 -> Red
                img[..., RGB_channel] = img_bin
            else:
                if k == 1 and self.img_reg is not None:
                    img = self.img_reg

        if k == 2:
            img = None
            img_0, img_1 = self.imgs

            if img_0 is not None and img_1 is not None:

                if mode == "Gray":
                    if self.img_reg is not None:
                        img_1 = self.img_reg
                    img_0, img_1 = padding(img_0, img_1)
                    img = 0.5 * (img_0 + img_1)

                elif mode == "Binarized":
                    img_0, img_1 = self.imgs_bin
                    if img_0 is None:
                        img_0 = self.binarization_k(0)
                    if img_1 is None:
                        img_1 = self.binarization_k(1)
                    img_0, img_1 = padding(img_0, img_1)
                    img = np.zeros((img_0.shape[0], img_0.shape[1], 3))
                    img[img_1 * ~img_0, 0] = 1
                    img[img_0 * ~img_1, 1] = 1

                elif mode == "Matching (SIFT)":
                    if self.matches is not None:
                        img_0, img_1 = self.imgs
                        plot_matches(self.ax[k], img_0, img_1,
                                     self.keypoints[0], self.keypoints[1],
                                     self.matches,
                                     alignment='vertical')
                        self.ax[k].invert_yaxis()

        if img is not None:
            self.ax[k].imshow(img, origin='lower', cmap='gray')
        if patch is not None:
            self.ax[k].add_patch(patch)
        self.ax[k].set_title(title)
        self.mpl_panes[k].param.trigger('object')
        pn.io.push_notebook(self.mpl_panes[k])

    def update_plot_zoom(self):

        self.ax[3].clear()

        ax = self.ax[AXES_TITLES.index(self.view_modes[1])]
        imgs = ax.get_images()
        if len(imgs) > 0:
            self.ax[3].imshow(imgs[0].get_array(), origin='lower', cmap='gray')
        lines = ax.get_lines()
        for line in lines:
            self.ax[3].plot(line.get_xdata(), line.get_ydata(),
                            c=line.get_color(), ls=line.get_linestyle())

        self.mpl_panes[3].param.trigger('object')
        pn.io.push_notebook(self.mpl_panes[3])

    def update_threshold(self, k, event):
        """ Update the k-th 'thresholds' attribute """
        self.thresholds[k] = event.new
        self.binarization()

    def update_reverse(self, k, event):
        """ Update the k-th 'bin_inversions' attribute """
        self.bin_inversions[k] = event.new
        self.binarization()

    def update_registration_model(self, event):
        """ Update the 'registration_model' attribute """
        self.registration_model = event.new
        self.update_plot(2)
        self.update_plot_zoom()

    def update_save_fname(self, event):
        """ Update the 'save_fname' attributes """
        self.save_fname = event.new

    def update_reload_fname(self, event):
        """ Update the 'reload_fname' attributes """
        self.reload_fname = event.new

    def reinit(self, k):
        """ Reinitialize the k-th image"""
        self.cropping_areas[k] = None
        self.h_range_sliders[k].value = (0, 1)
        self.v_range_sliders[k].value = (0, 1)
        self.update_file(k, fnames=[self.fnames[k]])

    def cropping(self, k, show_only=False):
        """ Crop the k-th image"""
        if self.cropping_areas[k] is not None:
            msg = "ERROR: 2 consecutive crops are not allowed. "
            msg += "Please, REINIT the image\n"
            self.terminal.write(msg)
            return

        xmin, xmax = self.h_range_sliders[k].value
        ymin, ymax = self.v_range_sliders[k].value
        shape = self.imgs[k].shape
        imin, imax = int(ymin * shape[0]), int(ymax * shape[0])
        jmin, jmax = int(xmin * shape[1]), int(xmax * shape[1])
        rect = Rectangle((jmin, imin), jmax - jmin, imax - imin,
                         fc='none', ec='w')

        if show_only:
            self.update_plot(k, patch=rect)
        else:
            self.imgs[k] = self.imgs[k][imin:imax, jmin:jmax]
            self.cropping_areas[k] = (imin, imax, jmin, jmax)
            self.update_plot(k)

        self.update_plot(2)
        self.update_plot_zoom()

    def resizing(self):
        """ Resize the low resolution image from the high resolution image """
        if self.imgs[0] is None or self.imgs[1] is None:
            return

        img0, img1, success = interpolation(*self.imgs)

        if success:
            self.imgs = [img0, img1]
        else:
            shape0, shape1 = self.imgs[0].shape, self.imgs[1].shape
            msg = 'Differentiation between low and high resolution image can ' \
                  f' not be done from shapes {shape0} and {shape1}\n\n'
            self.terminal.write(msg)
            return

        if self.mode_auto:
            self.binarization()
        else:
            [self.update_plot(i) for i in range(3)]
            self.update_plot_zoom()

    def binarization_k(self, k):
        """ Binarize the k-th image """
        img_bin = self.imgs[k] > self.thresholds[k]
        if edges_trend(self.imgs[k]):
            img_bin = ~img_bin
        if self.bin_inversions[k]:
            img_bin = ~img_bin
        return img_bin

    def binarization(self):
        """ Binarize the images """
        self.imgs_bin = [self.binarization_k(0), self.binarization_k(1)]

        if self.mode_auto:
            self.registration()
        else:
            [self.update_plot(i) for i in range(3)]
            self.update_plot_zoom()

    def registration(self, registration_model=None):
        self.registration_calc(registration_model=registration_model)
        self.registration_apply()

    def registration_calc(self, registration_model=None):
        """ Calculate the transformation matrix 'tmat' and apply it """
        if registration_model in REG_MODELS:
            self.reg_models.value = registration_model

        if self.registration_model == 'StackReg':
            self.imgs_bin[1] = self.binarization_k(1)  # reinit
            self.tmat = STREG.register(*self.imgs_bin)

        elif self.registration_model == 'SIFT':
            out = sift(*self.imgs)
            self.tmat, self.keypoints, self.descriptors, self.matches = out

        else:
            raise IOError

        print(self.tmat)

    def registration_apply(self):
        """ Apply 'tmat' to the moving image """

        self.imgs_bin[0] = self.binarization_k(0)  # reinit
        self.imgs_bin[1] = self.binarization_k(1)  # reinit

        self.img_reg = warp(self.imgs[1], self.tmat, mode='constant',
                            cval=1, preserve_range=True, order=None)
        self.imgs_bin[1] = warp(self.imgs_bin[1], self.tmat, mode='constant',
                                cval=1, preserve_range=True, order=None)

        # score calculation
        mask = warp(np.ones_like(self.imgs_bin[1]), self.tmat, mode='constant',
                    cval=0, preserve_range=True, order=None)
        mismatch = np.logical_xor(*self.imgs_bin)
        mismatch[~mask] = 0
        score = 100 * (1. - np.sum(mismatch) / np.sum(mask))

        self.results[self.registration_model] = {'score': score,
                                                 'tmat': self.tmat}

        np.set_printoptions(precision=4)
        self.result_str.object = f'SCORE: {score:.1f} % \n\n {self.tmat}'
        np.set_printoptions(precision=None)

        [self.update_plot(i) for i in range(3)]
        self.update_plot_zoom()

    def translate(self, mode):
        """ Apply translation STEP in 'tmat' """
        if mode == 'up':
            self.tmat[1, 2] -= STEP
        elif mode == 'down':
            self.tmat[1, 2] += STEP
        elif mode == 'left':
            self.tmat[0, 2] += STEP
        elif mode == 'right':
            self.tmat[0, 2] -= STEP
        self.registration_apply()

    def rotate(self, angle):
        """ Apply rotation coefficients in 'tmat' """

        rotation_mat = np.array([[np.cos(angle), np.sin(angle), 0],
                                 [-np.sin(angle), np.cos(angle), 0],
                                 [0, 0, 1]])

        shape = self.imgs_bin[0].shape
        xc_rel, yc_rel = self.xc_rel.value, self.yc_rel.value
        transl_y, transl_x = int(xc_rel * shape[0]), int(yc_rel * shape[1])

        transl = np.array([[1, 0, -transl_x],
                           [0, 1, -transl_y],
                           [0, 0, 1]])
        inv_transl = np.array([[1, 0, transl_x],
                               [0, 1, transl_y],
                               [0, 0, 1]])

        self.tmat = self.tmat @ inv_transl @ rotation_mat @ transl
        self.registration_apply()

    def select_dir_result(self):
        dirname = filedialog.askdirectory()
        if dirname:
            self.dir_results = dirname

    def apply(self, dirname_res=None):
        """ Apply the transformations to a set of images """
        if self.fnames_tot[0] is None:
            self.terminal.write("ERROR: fixed images are not defined\n\n")
            return
        if self.fnames_tot[1] is None:
            self.terminal.write("ERROR: moving images are not defined\n\n")
            return

        n0, n1 = len(self.fnames_tot[0]), len(self.fnames_tot[1])
        if not (n0 == 1 or n0 != n1):
            msg = f"ERROR: fixed images should consist in 1 or {n1} files.\n"
            msg += f"{n0} has been given\n\n"
            self.terminal.write(msg)
            return

        if dirname_res is None:
            dirname_res = filedialog.askdirectory()
            if dirname_res is None:
                return
        dirname_res = Path(dirname_res)
        dirname_res0 = dirname_res / "fixed_images"
        dirname_res1 = dirname_res / "moving_images"
        dirname_res.mkdir(exist_ok=True)
        dirname_res0.mkdir(exist_ok=True)
        dirname_res1.mkdir(exist_ok=True)

        fnames_fixed = self.fnames_tot[0]
        fnames_moving = self.fnames_tot[1]

        mode_auto_save = self.mode_auto
        self.mode_auto = False

        for i, fname_moving in enumerate(fnames_moving):

            fname_fixed = fnames_fixed[0] if n0 == 1 else fnames_fixed[i]

            self.fnames = [fname_fixed, fname_moving]

            name0 = Path(fname_fixed).name
            name1 = Path(fname_moving).name

            self.terminal.write(f"{i + 1}/{n1} {name0} - {name1}: ")

            try:

                self.imgs[0] = iio.imread(fname_fixed)
                self.imgs[1] = iio.imread(fname_moving)

                if self.cropping_areas[0] is not None:
                    (imin, imax, jmin, jmax) = self.cropping_areas[0]
                    self.imgs[0] = self.imgs[0][imin: imax, jmin: jmax]

                if self.cropping_areas[1] is not None:
                    (imin, imax, jmin, jmax) = self.cropping_areas[1]
                    self.imgs[1] = self.imgs[1][imin: imax, jmin: jmax]

                self.resizing()
                if i == 0 or not self.fixed_reg:
                    self.registration_calc()
                self.registration_apply()

                if n0 != 1 or i == 0:
                    iio.imwrite(dirname_res0 / name0, self.imgs[0])
                iio.imwrite(dirname_res1 / name1, self.img_reg)

                score = self.results[self.registration_model]['score']
                self.terminal.write(f"OK - score : {score:.1f} %\n")

            except:
                self.terminal.write("FAILED\n")

        self.terminal.write("\n")
        self.mode_auto = mode_auto_save

    def save(self, fname=None):
        """ Save data in a .json file """
        if fname is not None:
            self.save_fname = fname
            self.save_input.value = fname

        if self.save_fname is not None:
            data = {}
            for key in KEYS:
                data.update({key: eval(f"self.{key}")})
            data.update({'tmat': self.tmat.tolist()})

            with open(self.save_fname, 'w', encoding='utf-8') as fid:
                json.dump(data, fid, ensure_ascii=False, indent=4)

    @staticmethod
    def reload(fname):
        """ Reload data from .json file and Return an App() object """
        if os.path.isfile(fname):
            with open(fname, 'r', encoding='utf-8') as fid:
                data = json.load(fid)

            app = App(fnames_fixed=data['fnames_fixed'],
                      fnames_moving=data['fnames_moving'],
                      thresholds=data['thresholds'],
                      bin_inversions=data['bin_inversions'],
                      mode_auto=data['mode_auto'])
            for key, value in data.items():
                setattr(app, key, value)

            app.tmat = np.asarray(app.tmat)
            app.reload_fname = fname

            return app
        else:
            return None


if __name__ == "__main__":
    my_app = App()
    my_app.window.show()
