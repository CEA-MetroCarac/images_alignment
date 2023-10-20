"""
Application for images alignment
"""
import os
import glob
from copy import deepcopy
from pathlib import Path
import json
import panel as pn
import numpy as np
import imageio.v3 as iio
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from pystackreg import StackReg
from skimage.transform import warp
from skimage.color import rgba2rgb, rgb2gray
from scipy.interpolate import RegularGridInterpolator

AXES_TITLES = ['Moving image', 'Fixed image', 'Fused']
FLOW_MODES = ['Flow Auto', 'Iterative']
VIEW_MODES = ['Difference', 'Overlay']
STREG = StackReg(StackReg.SCALED_ROTATION)
STEP = 1  # translation increment
ANGLE = np.deg2rad(1)  # rotation angular increment
COEF1, COEF2 = 0.99, 1.01  # scaling coefficient

KEYS = ['fnames', 'thresholds', 'bin_inversions', 'mode_auto', 'tmat',
        'input_dirpath', 'output_dirpath']

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
        return img1, img2
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

    def __init__(self, fnames=None, thresholds=None, bin_inversions=None,
                 mode_auto=True):

        self.fnames = fnames or [None, None]
        self.thresholds = thresholds or [0.5, 0.5]
        self.bin_inversions = bin_inversions or [False, False]
        self.mode_auto = mode_auto

        self.window = None
        self.ax = [None, None, None]
        self.view_mode = VIEW_MODES[0]
        self.imgs = [None, None]
        self.cropping_areas = [None, None]
        self.imgs_bin = [None, None]
        self.tmat = np.identity(3)
        self.input_dirpath = None
        self.output_dirpath = None
        self.input_fnames = None
        self.save_fname = None
        self.reload_fname = None

        self.mpl_panes = [None, None, None]
        self.result_str = pn.pane.Str(self.tmat)
        self.terminal = pn.widgets.Terminal(align='center',
                                            options={"theme": {
                                                'background': '#F3F3F3',
                                                'foreground': '#000000'}})

        figs = [Figure(figsize=(4, 4)),
                Figure(figsize=(4, 4)),
                Figure(figsize=(8, 8))]

        for k in range(3):
            self.ax[k] = figs[k].subplots()
            self.ax[k].set_title(AXES_TITLES[k])
            self.ax[k].autoscale(tight=True)
            self.mpl_panes[k] = pn.pane.Matplotlib(figs[k], dpi=80, tight=True)

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
            file_input.value = self.fnames[k]
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
        self.register_button.on_click(lambda event: self.registration_auto())

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

        self.result_text = pn.pane.Str('')

        view_mode = pn.widgets.RadioButtonGroup(options=VIEW_MODES,
                                                button_style='outline',
                                                button_type='primary',
                                                value=self.view_mode)
        view_mode.param.watch(self.update_view_mode, 'value')

        apply_button = pn.widgets.Button(name='APPLY')
        apply_button.on_click(lambda event: self.apply())

        save_button = pn.widgets.Button(name='SAVE')
        save_button.on_click(lambda event: self.save())
        self.save_input = pn.widgets.FileInput(accept='.json')
        self.save_input.param.watch(self.update_save_fname, 'value')

        reload_button = pn.widgets.Button(name='RELOAD')
        reload_button.on_click(
            lambda _, fname=self.reload_fname: self.reload(fname=fname))
        reload_input = pn.widgets.FileInput(accept='.json')
        reload_input.param.watch(self.update_reload_fname, 'value')

        text = ['MOVING IMAGE', 'FIXED IMAGE']
        h_label = pn.pane.Str("H:")
        v_label = pn.pane.Str("V:")
        img_boxes = []
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
            img_boxes.append(img_box)

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

        proc_box = pn.Column(mode_auto_check,
                             pn.Row(self.resizing_button,
                                    self.binarize_button,
                                    self.register_button),
                             pn.Row(transl_box, pn.Spacer(width=30), rot_box),
                             self.result_str)
        proc_box = pn.WidgetBox(proc_box_title, proc_box, margin=(5, 0),
                                align='center', width=350)

        view_box_title = pn.pane.Markdown("**VIEW MODE**")
        view_box = pn.WidgetBox(view_box_title, view_mode, margin=(5, 0),
                                align='center', width=350)

        appl_box_title = pn.pane.Markdown("**APPLICATION**")

        self.input_dirpath_widget = pn.widgets.TextInput(width=600)
        self.input_dirpath_widget.param.watch(self.update_input_dirpath,
                                              'value')
        self.output_dirpath_widget = pn.widgets.TextInput(width=600)
        self.output_dirpath_widget.param.watch(self.update_output_dirpath,
                                               'value')
        row_input_dirpath = pn.Row(pn.pane.Markdown("INPUT Directory :"),
                                   self.input_dirpath_widget)
        row_output_dirpath = pn.Row(pn.pane.Markdown("OUTPUT Directory :"),
                                    self.output_dirpath_widget)

        appl_box = pn.WidgetBox(appl_box_title, row_input_dirpath,
                                row_output_dirpath, apply_button,
                                margin=(5, 20), sizing_mode='stretch_width')

        col1 = pn.Column(img_boxes[0],
                         img_boxes[1],
                         proc_box,
                         view_box,
                         width=380, height=850, auto_scroll_limit=100)

        col2 = pn.Column(pn.Row(self.mpl_panes[0], self.mpl_panes[1]),
                         self.mpl_panes[2])

        col3 = pn.Column(appl_box,
                         self.terminal,
                         pn.Row(save_button, self.save_input),
                         pn.Row(reload_button, reload_input))

        self.window = pn.Row(col1, col2, col3)
        self.update_disabled()

    def update_view_mode(self, event):
        """ Update the 'view_mode' attribute and replot """
        self.view_mode = event.new
        self.update_plot(2, binary=True)

    def update_mode_auto(self, event):
        """ Update the 'mode_auto' attribute """
        self.mode_auto = event.new == FLOW_MODES[0]
        self.update_disabled()

    def update_disabled(self):
        """ Change the disabled status of some buttons """
        self.resizing_button.disabled = self.mode_auto
        self.binarize_button.disabled = self.mode_auto
        self.register_button.disabled = self.mode_auto

    def update_file(self, k, fname):
        """ Load the k-th image file """
        try:
            img = iio.imread(fname)
        except Exception as _:
            self.terminal.write(f"Failed to load {fname}\n\n")
            return

        # image normalization in range [0, 1]
        self.imgs[k] = image_normalization(gray_conversion(img))

        # background uniformization
        if edges_trend(self.imgs[k]):
            self.imgs[k] = 1. - self.imgs[k]

        if self.bin_inversions[k]:
            self.imgs[k] = 1. - self.imgs[k]

        if self.mode_auto:
            self.resizing()
        else:
            self.update_plot(k, binary=False)

    def update_plot(self, k, binary=True, patch=None):
        """ Update the k-th ax """

        self.ax[k].clear()
        self.ax[k].set_title(AXES_TITLES[k])

        if patch is not None:
            self.ax[k].add_patch(patch)

        if not binary:
            img = self.imgs[k]

        elif k == 2:
            if self.view_mode == "Difference":
                img_0_bin, img_1_bin = self.imgs_bin
                img = np.zeros((img_0_bin.shape[0], img_0_bin.shape[1], 3))
                img[img_0_bin * ~img_1_bin, 0] = 1
                img[img_1_bin * ~img_0_bin, 1] = 1
            else:
                img_0, img_1 = self.imgs
                img_0_reg = warp(img_0, self.tmat, mode='constant',
                                 cval=1, preserve_range=True, order=None)
                img = 0.5 * (img_0_reg + img_1)

        else:
            img_bin = self.imgs_bin[k]
            shape = img_bin.shape
            img = np.zeros((shape[0], shape[1], 3))
            img[..., k] = img_bin

        self.ax[k].imshow(img, origin='lower', cmap='gray')
        self.mpl_panes[k].param.trigger('object')
        pn.io.push_notebook(self.mpl_panes[k])

    def update_threshold(self, k, event):
        """ Update the k-th 'thresholds' attribute """
        self.thresholds[k] = event.new
        self.binarization()

    def update_reverse(self, k, event):
        """ Update the k-th 'bin_inversions' attribute """
        self.bin_inversions[k] = event.new
        self.binarization()

    def update_input_dirpath(self, event):
        """ Update the 'input_dirpath' attributes and get the related files """
        dirpath = event.new
        if not os.path.isdir(dirpath):
            self.terminal.write(f"ERROR: {dirpath} is not a directory\n\n")
            return
        else:
            fnames = glob.glob(os.path.join(dirpath, "*"))
            self.terminal.write(f"{len(fnames) + 1} files have been loaded\n\n")
            self.input_dirpath = dirpath
            self.input_fnames = fnames

    def update_output_dirpath(self, event):
        """ Update the 'output_dirpath' attributes """
        dirpath = event.new
        try:
            os.makedirs(dirpath, exist_ok=True)
            self.output_dirpath = dirpath
        except:
            self.terminal.write(f"ERROR: Failed to create {dirpath}\n\n")

    def update_save_fname(self, event):
        """ Update the 'save_fname' attributes """
        self.save_fname = event.new

    def update_reload_fname(self, event):
        """ Update the 'reload_fname' attributes """
        self.reload_fname = event.new

    def reinit(self, k):
        """ Reinitialize the k-th image"""
        self.h_range_sliders[k].value = (0, 1)
        self.v_range_sliders[k].value = (0, 1)
        self.cropping_areas[k] = None
        self.update_file(k, fname=self.fnames[k])

    def cropping(self, k, show_only=False):
        """ Crop the k-th image"""
        if self.cropping_areas[k] is not None:
            msg = "ERROR: 2 consecutive crops are not allowed\n"
            msg += "please, REINIT the image\n\n"
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
            self.update_plot(k, binary=False)

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
            self.update_plot(0, binary=False)
            self.update_plot(1, binary=False)

    def binarization(self):
        """ Binarize the images """
        self.imgs_bin = [self.imgs[0] > self.thresholds[0],
                         self.imgs[1] > self.thresholds[1]]

        self.update_plot(1)

        if self.mode_auto:
            self.registration_auto()
        else:
            self.update_plot(0)

    def registration_auto(self):
        """ Calculate 'tmat' from pystackreg and apply it """
        self.imgs_bin[0] = self.imgs[0] > self.thresholds[0]  # reinit
        self.tmat = STREG.register(*self.imgs_bin[::-1])
        self.registration()

    def registration(self):
        """ Apply 'tmat' to the binarized moving image """
        np.set_printoptions(precision=4)
        self.result_str.object = self.tmat
        np.set_printoptions(precision=None)

        self.imgs_bin[0] = self.imgs[0] > self.thresholds[0]  # reinit
        self.imgs_bin[0] = warp(self.imgs_bin[0], self.tmat, mode='constant',
                                cval=1, preserve_range=True, order=None)

        self.update_plot(0)
        self.update_plot(2)

    def translate(self, mode):
        """ Apply translation STEP in 'tmat' """
        print(mode)
        if mode == 'up':
            self.tmat[1, 2] -= STEP
        elif mode == 'down':
            self.tmat[1, 2] += STEP
        elif mode == 'left':
            self.tmat[0, 2] += STEP
        elif mode == 'right':
            self.tmat[0, 2] -= STEP
        self.registration()

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
        self.registration()

    def apply(self):
        """ Apply the transformation matrix 'tmat' to a set of images """
        if self.imgs[1] is None:
            self.terminal.write("ERROR: Fixed image is not defined\n\n")
            return

        # save fixed image (in case of cropping)
        name = Path(self.fnames[1]).name
        iio.imwrite(os.path.join(self.output_dirpath, name), self.imgs[1])

        nfnames = len(self.input_fnames)
        for i, fname in enumerate(self.input_fnames):
            name = Path(fname).name
            self.terminal.write(f"{i + 1}/{nfnames} {name}: ")

            try:
                img = iio.imread(fname)
            except:
                self.terminal.write("Failed to decode image from file\n")
                continue

            try:
                if self.cropping_areas[0] is not None:
                    imin, imax, jmin, jmax = self.cropping_areas[0]
                    img = img[imin: imax, jmin: jmax]
                img_int, _, _ = interpolation(img, self.imgs[1])
                img_reg = warp(img_int, self.tmat, mode='constant', cval=0,
                               preserve_range=True, order=None)
                iio.imwrite(os.path.join(self.output_dirpath, name), img_reg)
                self.terminal.write("OK\n")
            except:
                self.terminal.write("FAILED\n")

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

            app = App(fnames=data['fnames'],
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