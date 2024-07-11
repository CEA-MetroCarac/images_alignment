"""
Application for images registration
"""
from pathlib import Path
import tempfile
import panel as pn
import numpy as np
import param
import imageio.v3 as iio
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from bokeh.plotting import figure as Figure_bokeh
from bokeh.models import DataRange1d, LinearColorMapper
from skimage.feature import plot_matches
from panel.widgets import (FileInput, Button, RadioButtonGroup, FloatSlider,
                           FloatInput, Checkbox, TextInput, Terminal)

from images_alignment import ImagesAlign, REG_MODELS, KEYS
from images_alignment.utils import padding

AXES_TITLES = ['Fixed image', 'Moving image', 'Combined image', None]
FLOW_MODES = ['Flow Auto', 'Iterative']
VIEW_MODES = [['Gray', 'Binarized', 'Matching (SIFT)'], AXES_TITLES[:3]]

COLORS = [(0, 1, 0), (0, 0, 0), (1, 0, 0)]  # Green -> Black -> Red
CMAP_BINARIZED = LinearSegmentedColormap.from_list('GreenBlackRed', COLORS, N=3)
CMAP_BINARIZED_BOKEH = LinearColorMapper(low=-1, high=1,
                                         palette=["#00FF00", "black", "red"])

pn.extension('terminal', inline=True)
pn.pane.Str.align = 'center'
pn.pane.Markdown.align = 'center'
pn.pane.Matplotlib.align = 'center'
pn.Row.align = 'center'
pn.Column.align = 'center'
Button.align = 'center'
RadioButtonGroup.align = 'center'
FloatSlider.align = 'center'
Checkbox.align = 'center'
TextInput.align = 'center'
FileInput.align = 'center'
Terminal.align = 'center'


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
                 thresholds=None, bin_inversions=None, mode_auto=False):

        self.window = None
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ax = [None, None, None, None]
        self.view_modes = ['Gray', 'Fixed image']
        self.mpl_panes = [None, None, None, None]
        self.result_str = pn.pane.Str(None, styles={'text-align': 'center'})
        self.terminal = Terminal(height=100,
                                 sizing_mode='stretch_width',
                                 options={"theme": {'background': '#F3F3F3',
                                                    'foreground': '#000000'}})

        self.model = ImagesAlign(fnames_fixed=fnames_fixed,
                                 fnames_moving=fnames_moving,
                                 thresholds=thresholds,
                                 bin_inversions=bin_inversions,
                                 mode_auto=mode_auto,
                                 terminal=self.terminal)

        self.figs = [Figure(figsize=(4, 4)),
                     Figure(figsize=(4, 4)),
                     Figure(figsize=(4, 4)),
                     Figure_bokeh(x_range=DataRange1d(range_padding=0),
                                  y_range=DataRange1d(range_padding=0),
                                  width=300, height=300, match_aspect=True,
                                  sizing_mode='stretch_both')]

        for k in range(3):
            self.ax[k] = self.figs[k].subplots()
            self.ax[k].set_title(AXES_TITLES[k])
            self.ax[k].autoscale(tight=True)
            self.mpl_panes[k] = pn.pane.Matplotlib(self.figs[k],
                                                   dpi=80, tight=True,
                                                   align='center',
                                                   sizing_mode='stretch_both')

        self.mpl_panes[3] = pn.pane.Bokeh(self.figs[3])

        file_inputs = []
        reinit_buttons = []
        crop_buttons = []
        self.thresh_sliders = []
        self.reverse_checks = []
        for k in range(2):
            file_input = FileInput(accept='image/*', multiple=True)
            file_input.param.watch(
                lambda event, k=k: self.update_files(k, event), 'filename')
            file_inputs.append(file_input)

            reinit_button = Button(name='REINIT')
            reinit_button.on_click(lambda _, k=k: self.update_reinit(k))
            reinit_buttons.append(reinit_button)

            crop_button = Button(name='CROP FROM ZOOM')
            crop_button.on_click(lambda _, k=k: self.update_cropping(k))
            crop_buttons.append(crop_button)

            thresh_slider = FloatSlider(name='threshold ', step=0.01,
                                        value=self.model.thresholds[k],
                                        bar_color='#0072b5',
                                        width=150)
            thresh_slider.param.watch(
                lambda event, k=k: self.update_threshold(k, event), 'value')
            self.thresh_sliders.append(thresh_slider)

            reverse_check = Checkbox(name='Reversed',
                                     value=self.model.bin_inversions[k],
                                     height=0)
            reverse_check.param.watch(
                lambda event, k=k: self.update_reverse(k, event), 'value')
            self.reverse_checks.append(reverse_check)

        value = FLOW_MODES[not self.model.mode_auto]
        mode_auto_check = RadioButtonGroup(options=FLOW_MODES,
                                           button_style='outline',
                                           button_type='primary',
                                           value=value)
        mode_auto_check.param.watch(self.update_mode_auto, 'value')

        self.resizing_button = Button(name='RESIZING', margin=2)
        self.resizing_button.on_click(lambda _: self.update_resizing())

        self.binarize_button = Button(name='BINARIZATION', margin=2)
        self.binarize_button.on_click(lambda _: self.update_binarization())

        self.register_button = Button(name='REGISTRATION', margin=2)
        self.register_button.on_click(lambda _: self.update_registration())

        value = self.model.registration_model
        self.reg_models = RadioButtonGroup(options=REG_MODELS,
                                           button_style='outline',
                                           button_type='primary',
                                           value=value)
        self.reg_models.param.watch(self.update_registration_model, 'value')

        transl_up_but = Button(name='▲')
        transl_down_but = Button(name='▼')
        transl_left_but = Button(name='◄', margin=5)
        transl_right_but = Button(name='►', margin=5)
        transl_up_but.on_click(lambda _, mode='up':
                               self.model.translate(mode))
        transl_down_but.on_click(lambda _, mode='down':
                                 self.model.translate(mode))
        transl_left_but.on_click(lambda _, mode='left':
                                 self.model.translate(mode))
        transl_right_but.on_click(lambda _, mode='right':
                                  self.model.translate(mode))

        rot_clock_button = Button(name='↻')
        rot_anticlock_button = Button(name='↺')
        rot_clock_button.on_click(lambda _: self.model.rotate())
        rot_anticlock_button.on_click(lambda _: self.model.rotate(reverse=True))

        self.xc_rel = FloatInput(value=0.5, width=60)
        self.yc_rel = FloatInput(value=0.5, width=60)

        view_mode = RadioButtonGroup(options=VIEW_MODES[0],
                                     button_style='outline',
                                     button_type='primary',
                                     value=self.view_modes[0],
                                     align='center')
        view_mode.param.watch(self.update_view_mode, 'value')

        view_mode_zoom = RadioButtonGroup(options=VIEW_MODES[1],
                                          button_style='outline',
                                          button_type='primary',
                                          value=self.view_modes[1],
                                          align='center')
        view_mode_zoom.param.watch(self.update_view_mode_zoom, 'value')

        select_dir_button = Button(name='SELECT DIR. RESULT')
        select_dir_button.on_click(lambda _: self.select_dir_result())

        apply_button = Button(name='APPLY & SAVE')
        apply_button.on_click(lambda _: self.model.apply())

        fixed_reg_check = Checkbox(name='Fixed registration',
                                   value=self.model.fixed_reg)

        save_button = Button(name='SAVE MODEL')
        save_button.on_click(lambda _: self.model.save_model())

        reload_button = Button(name='RELOAD MODEL')
        reload_button.on_click(lambda _: self.reload_model())

        boxes = []

        text = ['FIXED IMAGE', 'MOVING IMAGE']
        for k in range(2):
            img_box = pn.WidgetBox(pn.pane.Markdown(f"**{text[k]}**"),
                                   file_inputs[k],
                                   pn.Row(reinit_buttons[k], crop_buttons[k]),
                                   pn.Row(self.thresh_sliders[k],
                                          self.reverse_checks[k]),
                                   margin=(5, 5), width=350)
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
        proc_box = pn.WidgetBox(proc_box_title, proc_box, margin=(5, 5),
                                width=350)
        boxes.append(proc_box)

        appl_box_title = pn.pane.Markdown("**APPLICATION**")

        appl_box = pn.WidgetBox(appl_box_title,
                                pn.Row(apply_button, fixed_reg_check),
                                pn.Row(save_button, reload_button),
                                margin=(5, 5), width=350)

        col1 = pn.Column(*boxes, appl_box, width=350)

        col2 = pn.Column(view_mode,
                         self.mpl_panes[0],
                         self.mpl_panes[1],
                         self.mpl_panes[2], width=350)

        col3 = pn.Column(view_mode_zoom,
                         self.mpl_panes[3],
                         self.terminal,
                         sizing_mode='stretch_width')

        self.window = pn.Row(col1, col2, col3, sizing_mode='stretch_both')
        self.update_disabled()

    def get_selected_figure_index(self):
        """ Return the index of the selected figure """
        return AXES_TITLES.index(self.view_modes[1])

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
        self.model.mode_auto = event.new == FLOW_MODES[0]
        self.update_disabled()

    def update_disabled(self):
        """ Change the disabled status of some buttons """
        self.resizing_button.disabled = self.model.mode_auto
        self.binarize_button.disabled = self.model.mode_auto
        self.register_button.disabled = self.model.mode_auto

    def update_files(self, k, fnames):
        """ Load the k-th image files """
        # make a local copy (in the host) of the images issued from pn.FileInput
        if isinstance(fnames, param.parameterized.Event):
            dirname = Path(self.tmpdir.name)
            fnames_ = []
            for fname, value in zip(fnames.new, fnames.obj.value):
                fname_ = dirname / fname
                arr = iio.imread(value)
                iio.imwrite(fname_, arr)
                fnames_.append(fname_)
            fnames = fnames_

        self.model.load_files(k, fnames=fnames)

        self.update_plot(k)
        self.update_plot(2)

        if self.model.mode_auto:
            self.model.resizing()

    def update_plot(self, k, patch=None):
        """ Update the k-th ax """
        self.ax[k].clear()

        img = None
        mode = self.view_modes[0]
        title = AXES_TITLES[k]

        if mode == 'Binarized':
            cmap, vmin, vmax = CMAP_BINARIZED, -1, 1
        else:
            cmap, vmin, vmax = 'gray', None, None

        if k in [0, 1]:
            img = self.model.imgs[k]
            if self.model.fnames[k] is not None:
                title += f" - {self.model.fnames[k].name}"

            if mode == 'Binarized':
                img_bin = self.model.imgs_bin[k]
                if img_bin is None:
                    img_bin = self.model.binarization_k(k)
                img = np.zeros_like(img_bin, dtype=int)
                img[img_bin] = 2 * k - 1
            else:
                if k == 1 and self.model.img_reg is not None:
                    img = self.model.img_reg

        if k == 2:
            img = None
            img_0, img_1 = self.model.imgs

            if img_0 is not None and img_1 is not None:

                if mode == "Gray":
                    if self.model.img_reg is not None:
                        img_1 = self.model.img_reg
                    img_0, img_1 = padding(img_0, img_1)
                    img = 0.5 * (img_0 + img_1)

                elif mode == "Binarized":
                    img_0, img_1 = self.model.imgs_bin
                    if img_0 is None:
                        img_0 = self.model.binarization_k(0)
                    if img_1 is None:
                        img_1 = self.model.binarization_k(1)
                    img_0, img_1 = padding(img_0, img_1)
                    img = np.zeros_like(img_0, dtype=int)
                    img[img_1 * ~img_0] = -1
                    img[img_0 * ~img_1] = 1

                elif mode == "Matching (SIFT)":
                    if self.model.matches is not None:
                        img_0, img_1 = self.model.imgs
                        plot_matches(self.ax[k], img_0, img_1,
                                     self.model.keypoints[0],
                                     self.model.keypoints[1],
                                     self.model.matches,
                                     alignment='vertical')
                        self.ax[k].invert_yaxis()

        if img is not None:
            self.ax[k].imshow(img, origin='lower',
                              cmap=cmap, vmin=vmin, vmax=vmax)
        if patch is not None:
            self.ax[k].add_patch(patch)
        self.ax[k].set_title(title)
        self.mpl_panes[k].param.trigger('object')
        pn.io.push_notebook(self.mpl_panes[k])

    def update_plot_zoom(self):
        """ Update the bokeh image """
        fig = self.figs[3]

        fig.renderers.clear()

        k = self.get_selected_figure_index()
        ax = self.ax[k]
        imgs = ax.get_images()
        if len(imgs) > 0:
            arr = imgs[0].get_array()
            if self.view_modes[0] == "Binarized":
                color_mapper = CMAP_BINARIZED_BOKEH
            else:
                color_mapper = LinearColorMapper(palette="Greys256")
            fig.image([arr], x=0, y=0,
                      dw=arr.shape[1], dh=arr.shape[0],
                      color_mapper=color_mapper)
            # fig.x_range.start, fig.x_range.end = 0, arr.shape[1]
            # fig.y_range.start, fig.y_range.end = 0, arr.shape[0]
            # fig.aspect_ratio = arr.shape[1] / arr.shape[0]

        lines = ax.get_lines()
        for line in lines:
            fig.line(x=line.get_xdata(), y=line.get_ydata(),
                     line_color=line.get_color(),
                     line_dash=line.get_linestyle())

        self.mpl_panes[3].param.trigger('object')
        pn.io.push_notebook(self.mpl_panes[3])

    def update_reinit(self, k):
        """ Reinit the k-th image """
        self.model.reinit(k)
        self.update_files(k, fnames=[self.model.fnames[k]])
        self.update_plot_zoom()

    def update_cropping(self, k):
        """ Update the cropping of the k-th image """
        x, y = self.figs[3].x_range, self.figs[3].y_range
        self.model.cropping_areas[k] = [int(y.start), int(y.end),
                                        int(x.start), int(x.end)]
        self.model.cropping(k)

        if not self.model.mode_auto:
            [self.update_plot(i) for i in range(3)]
            self.update_plot_zoom()

    def update_resizing(self):
        """ Resize the images """
        self.model.resizing()

        if not self.model.mode_auto:
            [self.update_plot(i) for i in range(3)]
            self.update_plot_zoom()

    def update_threshold(self, k, event):
        """ Update the k-th 'thresholds' attribute """
        self.model.thresholds[k] = event.new
        self.update_binarization()

    def update_reverse(self, k, event):
        """ Update the k-th 'bin_inversions' attribute """
        self.model.bin_inversions[k] = event.new
        self.update_binarization()

    def update_binarization(self):
        """ Binarize the images """
        self.model.binarization()

        if not self.model.mode_auto:
            [self.update_plot(i) for i in range(3)]
            self.update_plot_zoom()

    def update_registration_model(self, event):
        """ Update the 'registration_model' attribute """
        self.model.registration_model = event.new

    def update_registration(self):
        """ Apply registration """
        self.model.registration()
        self.update_plot(2)
        self.update_plot_zoom()

    def reload_model(self):
        """ Reload model """
        model_reloaded = self.model.reload_model()
        for key in KEYS:
            setattr(self.model, key, eval(f"model_reloaded.{key}"))

        for k in range(2):
            self.thresh_sliders[k].value = self.model.thresholds[k]
            self.reverse_checks[k].value = self.model.bin_inversions[k]


if __name__ == "__main__":
    my_app = App()
    my_app.window.show()
