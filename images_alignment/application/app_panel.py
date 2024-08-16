"""
Application for images registration
"""
from pathlib import Path
import tempfile
import panel as pn
import param
import imageio.v3 as iio
from matplotlib.figure import Figure
from bokeh.plotting import figure as Figure_bokeh
from bokeh.models import DataRange1d, LinearColorMapper
from panel.widgets import (FileInput, Button, RadioButtonGroup, FloatSlider,
                           FloatInput, Checkbox, TextInput, Terminal)

from images_alignment import ImagesAlign
from images_alignment import REG_MODELS, KEYS, AXES_NAMES, STEP, ANGLE, COEF

VIEW_MODES = ['Gray', 'Binarized', 'Matching (SIFT)']
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
        Images pathname to handle
    thresholds: iterable of 2 floats, optional
        Thresholds used to binarize the images
    bin_inversions: iterable of 2 bools, optional
        Activation keywords to reverse the image binarization
    """

    def __init__(self, fnames_fixed=None, fnames_moving=None,
                 thresholds=None, bin_inversions=None):

        self.window = None
        self.tmpdir = tempfile.TemporaryDirectory()
        self.view_mode = 'Gray'
        self.view_mode_bokeh = 'Fixed image'
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
                                 terminal=self.terminal)

        self.figs = [Figure(figsize=(4, 4)),
                     Figure(figsize=(4, 4)),
                     Figure(figsize=(4, 4)),
                     Figure_bokeh(x_range=DataRange1d(range_padding=0),
                                  y_range=DataRange1d(range_padding=0),
                                  width=300, height=300, match_aspect=True,
                                  sizing_mode='stretch_both')]

        for k in range(3):
            self.model.ax[k] = self.figs[k].subplots()  # model.ax overridden
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
            reinit_button.on_click(lambda _, k=k: self.reinit(k))
            reinit_buttons.append(reinit_button)

            crop_button = Button(name='CROP FROM ZOOM')
            crop_button.on_click(lambda _, k=k: self.cropping(k))
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

        resizing_button = Button(name='RESIZING', margin=2)
        resizing_button.on_click(lambda _: self.resizing())

        binarization_button = Button(name='BINARIZATION', margin=2)
        binarization_button.on_click(lambda _: self.binarization())

        registration_button = Button(name='REGISTRATION', margin=2)
        registration_button.on_click(lambda _: self.registration())

        value = self.model.registration_model
        reg_models = RadioButtonGroup(options=REG_MODELS,
                                      button_style='outline',
                                      button_type='primary',
                                      value=value)
        reg_models.param.watch(self.update_registration_model, 'value')

        transl_up_but = Button(name='▲')
        transl_down_but = Button(name='▼')
        transl_left_but = Button(name='◄', margin=5)
        transl_right_but = Button(name='►', margin=5)
        transl_up_but.on_click(lambda _, mode='up': self.translate(mode))
        transl_down_but.on_click(lambda _, mode='down': self.translate(mode))
        transl_left_but.on_click(lambda _, mode='left': self.translate(mode))
        transl_right_but.on_click(lambda _, mode='right': self.translate(mode))

        rot_clock_button = Button(name='↻')
        rot_anticlock_button = Button(name='↺')
        rot_clock_button.on_click(lambda _: self.rotate())
        rot_anticlock_button.on_click(lambda _: self.rotate(clockwise=False))

        self.xc_rel = FloatInput(value=0.5, width=60)
        self.yc_rel = FloatInput(value=0.5, width=60)

        view_mode = RadioButtonGroup(options=VIEW_MODES,
                                     button_style='outline',
                                     button_type='primary',
                                     value=self.view_mode,
                                     align='center')
        view_mode.param.watch(self.update_view_mode, 'value')

        view_mode_bokeh = RadioButtonGroup(options=AXES_NAMES,
                                           button_style='outline',
                                           button_type='primary',
                                           value=self.view_mode_bokeh,
                                           align='center')
        view_mode_bokeh.param.watch(self.update_view_mode_bokeh, 'value')

        select_dir_button = Button(name='SELECT DIR. RESULT')
        select_dir_button.on_click(lambda _: self.set_dirname_res())

        fixed_reg_check = Checkbox(name='Fixed registration',
                                   value=self.model.fixed_reg)

        apply_button = Button(name='APPLY TO ALL')
        apply_button.on_click(lambda _: self.apply_to_all())

        up_button = Button(name='▲')
        down_button = Button(name='▼')
        up_button.on_click(lambda _, mode='up': self.change_images(mode))
        down_button.on_click(lambda _, mode='down': self.change_images(mode))

        save_button = Button(name='SAVE')
        save_button.on_click(lambda _: self.model.save_model())

        reload_button = Button(name='RELOAD')
        reload_button.on_click(lambda _: self.reload_model())

        boxes = []

        text = ['FIXED IMAGE', 'MOVING IMAGE']
        for k in range(2):
            box = pn.WidgetBox(pn.pane.Markdown(f"**{text[k]}**"),
                               file_inputs[k],
                               pn.Row(reinit_buttons[k], crop_buttons[k]),
                               pn.Row(self.thresh_sliders[k],
                                      self.reverse_checks[k]),
                               margin=(5, 5), width=350)
            boxes.append(box)

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

        box = pn.WidgetBox(pn.pane.Markdown("**IMAGES PRE-PROCESSING**"),
                           pn.Row(resizing_button,
                                  binarization_button,
                                  registration_button),
                           reg_models,
                           margin=(5, 5), width=350)
        boxes.append(box)

        box = pn.WidgetBox(pn.pane.Markdown("**APPLICATION**"),
                           select_dir_button,
                           pn.Row(fixed_reg_check, apply_button),
                           pn.Row(up_button, down_button,
                                  save_button, reload_button),
                           margin=(5, 5), width=350)
        boxes.append(box)

        box = pn.WidgetBox(pn.pane.Markdown("**REGISTRATION**"),
                           self.result_str,
                           pn.Row(transl_box, pn.Spacer(width=30), rot_box),
                           margin=(5, 5), width=350)
        boxes.append(box)

        col1 = pn.Column(*boxes, width=350)

        col2 = pn.Column(view_mode,
                         self.mpl_panes[0],
                         self.mpl_panes[1],
                         self.mpl_panes[2], width=350)

        col3 = pn.Column(view_mode_bokeh,
                         self.mpl_panes[3],
                         self.terminal,
                         sizing_mode='stretch_width')

        self.window = pn.Row(col1, col2, col3, sizing_mode='stretch_both')
        self.update_result_str()

    def get_selected_figure_index(self):
        """ Return the index of the selected figure """
        return AXES_NAMES.index(self.view_mode_bokeh)

    def update_view_mode(self, event):
        """ Update the 'view_mode' attribute and replot """
        self.view_mode = event.new
        self.update_plots()

    def update_view_mode_bokeh(self, event):
        """ Update the 'view_mode' attribute and replot """
        self.view_mode_bokeh = event.new
        self.update_plot_bokeh()

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
        self.update_plots()

    def update_plots(self):
        """ Update all the plots """
        for k in range(3):
            self.update_plot_k(k)
        self.update_plot_bokeh()

    def update_plot_k(self, k):
        """ Update the k-th ax """
        self.model.plot_k(k)
        self.mpl_panes[k].param.trigger('object')
        pn.io.push_notebook(self.mpl_panes[k])

    def update_plot_bokeh(self):
        """ Update the bokeh image """
        fig = self.figs[3]

        fig.renderers.clear()

        k = self.get_selected_figure_index()
        ax = self.model.ax[k]
        imgs = ax.get_images()
        if len(imgs) > 0:
            arr = imgs[0].get_array()
            if self.view_mode == "Binarized":
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
            color = (255 * line.get_color()).astype(int)
            hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
            fig.line(x=line.get_xdata(), y=line.get_ydata(),
                     line_color=hex_color, line_width=2)

        self.mpl_panes[3].param.trigger('object')
        pn.io.push_notebook(self.mpl_panes[3])

    def update_threshold(self, k, event):
        """ Update the k-th 'thresholds' attribute """
        self.model.thresholds[k] = event.new
        self.binarization()

    def update_reverse(self, k, event):
        """ Update the k-th 'bin_inversions' attribute """
        self.model.bin_inversions[k] = event.new
        self.binarization()

    def update_registration_model(self, event):
        """ Update the 'registration_model' attribute """
        self.model.registration_model = event.new

    def update_result_str(self):
        """ Update the result_str object """
        score, tmat = self.model.score, self.model.tmat
        self.result_str.object = f'SCORE: {score:.1f} % \n\n {tmat}'

    def update(self):
        self.update_plot_k(2)
        self.update_plot_bokeh()
        self.update_result_str()

    def reinit(self, k):
        """ Reinit the k-th image """
        self.update_files(k, fnames=[self.model.fnames[k]])
        self.update_plot_bokeh()

    def cropping(self, k):
        """ Update the cropping of the k-th image """
        x, y = self.figs[3].x_range, self.figs[3].y_range
        self.model.cropping_areas[k] = [int(y.start), int(y.end),
                                        int(x.start), int(x.end)]
        self.model.cropping(k)
        self.update_plots()

    def resizing(self):
        """ Resize the images """
        self.model.resizing()
        self.update_plots()

    def binarization(self):
        """ Binarize the images """
        self.model.binarization()
        self.update_plots()

    def registration(self):
        """ Apply registration """
        self.model.registration()
        self.update()

    def translate(self, mode):
        """ Translate the moving image """
        self.model.translate(mode=mode)
        self.update()

    def rotate(self, clockwise=True):
        """ Rotate the moving image """
        xc_rel, yc_rel = self.xc_rel.value, self.yc_rel.value
        angle = -ANGLE if clockwise else ANGLE
        self.model.rotate(angle, xc_rel=xc_rel, yc_rel=yc_rel)
        self.update()

    def rescale(self, mode):
        """ Rescale the moving image """
        xc_rel, yc_rel = self.xc_rel.value, self.yc_rel.value
        coef = COEF is mode == 'up'
        self.model.rescale(coef, xc_rel=xc_rel, yc_rel=yc_rel)
        self.update()

    def set_dirname_res(self):
        """ Select the dirname where to save the results """
        self.model.set_dirname_res()

    def apply_to_all(self, dirname_res=None):
        """ Apply the model to all the set of moving images """
        self.model.apply_to_all(dirname_res=dirname_res)
        self.update_result_str()

    def change_images(self, mode):
        """ Select next or previous images of the set """
        ind0 = self.model.fnames_tot[0].index(self.model.fnames[0])
        ind1 = self.model.fnames_tot[1].index(self.model.fnames[1])
        if mode == 'up':
            ind0_new = max(0, ind0 - 1)
            ind1_new = max(0, ind1 - 1)
        else:
            ind0_new = min(len(self.model.fnames_tot[0]) - 1, ind0 + 1)
            ind1_new = min(len(self.model.fnames_tot[1]) - 1, ind1 + 1)

        def load_image(k, fname):
            self.model.load_image(k, fname)
            self.model.is_cropped[k] = False
            self.model.cropping(k, area=self.model.cropping_areas[k])
            self.model.resizing()
            if k == 1 and self.model.dirname_res[k] is not None:
                fname_res = self.model.dirname_res[k] / fname.name
                if fname_res.exists():
                    self.model.img_reg = iio.imread(fname_res)

        if ind0_new != ind0:
            load_image(0, self.model.fnames_tot[0][ind0_new])
        if ind1_new != ind1:
            load_image(1, self.model.fnames_tot[1][ind1_new])

        self.update_plots()

    def reload_model(self):
        """ Reload model """
        model_reloaded = self.model.reload_model()
        for key in KEYS:
            setattr(self.model, key, eval(f"model_reloaded.{key}"))

        for k in range(2):
            self.thresh_sliders[k].value = self.model.thresholds[k]
            self.reverse_checks[k].value = self.model.bin_inversions[k]


def launcher():
    """ Launch the panel application """
    app = App()
    app.window.show()


if __name__ == "__main__":
    launcher()
