"""
Application for images registration
"""
from copy import deepcopy
import panel as pn
import numpy as np
import imageio.v3 as iio
from matplotlib.figure import Figure
from bokeh.plotting import figure as Figure_bokeh
from bokeh.models import DataRange1d
from skimage.feature import plot_matches

from images_alignment import ImagesAlign, REG_MODELS
from images_alignment.utils import (gray_conversion, image_normalization,
                                    padding)

AXES_TITLES = ['Fixed image', 'Moving image', 'Combined image', None]
FLOW_MODES = ['Flow Auto', 'Iterative']
VIEW_MODES = [['Gray', 'Binarized', 'Matching (SIFT)'], AXES_TITLES[:3]]

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


class App(ImagesAlign):
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

        super().__init__(fnames_fixed=fnames_fixed, fnames_moving=fnames_moving,
                         thresholds=thresholds, bin_inversions=bin_inversions,
                         mode_auto=mode_auto)

        self.window = None
        self.ax = [None, None, None, None]
        self.view_modes = ['Gray', 'Fixed image']
        self.mpl_panes = [None, None, None, None]
        self.result_str = pn.pane.Str(None, align='center',
                                      styles={'text-align': 'center'})
        self.terminal = pn.widgets.Terminal(align='center',
                                            height=100,
                                            sizing_mode='stretch_width',
                                            options={"theme": {
                                                'background': '#F3F3F3',
                                                'foreground': '#000000'}})

        self.figs = [Figure(figsize=(4, 4)),
                     Figure(figsize=(4, 4)),
                     Figure(figsize=(4, 4)),
                     Figure_bokeh(x_range=DataRange1d(range_padding=0),
                                  y_range=DataRange1d(range_padding=0),
                                  width=300, height=300,
                                  sizing_mode='stretch_both')]

        for k in range(3):
            self.ax[k] = self.figs[k].subplots()
            self.ax[k].set_title(AXES_TITLES[k])
            self.ax[k].autoscale(tight=True)
            self.figs[k].canvas.mpl_connect('button_press_event',
                                            lambda _: self.plot_selection(k))

            self.mpl_panes[k] = pn.pane.Matplotlib(self.figs[k],
                                                   dpi=80, tight=True,
                                                   align='center',
                                                   sizing_mode='stretch_both')
        self.figs[3].match_aspect = True
        self.mpl_panes[3] = pn.panel(self.figs[3])

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
        rot_clock_button.on_click(lambda event: self.rotate())
        rot_anticlock_button.on_click(lambda event: self.rotate(reverse=True))

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

        self.figs[3].renderers.clear()

        ax = self.ax[AXES_TITLES.index(self.view_modes[1])]
        imgs = ax.get_images()
        if len(imgs) > 0:
            arr = imgs[0].get_array()
            self.figs[3].image([arr], x=0, y=0,
                               dw=arr.shape[1], dh=arr.shape[0])

        lines = ax.get_lines()
        for line in lines:
            self.figs[3].line(x=line.get_xdata(), y=line.get_ydata(),
                              line_color=line.get_color(),
                              line_dash=line.get_linestyle())

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


if __name__ == "__main__":
    my_app = App()
    my_app.window.show()
