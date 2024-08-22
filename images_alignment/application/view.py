"""
Class View attached to the application
"""
from tkinter import (Frame, LabelFrame, Label, Radiobutton, Scale,
                     Button, Checkbutton, Entry,
                     W, E, HORIZONTAL, DoubleVar, StringVar, BooleanVar)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from images_alignment.application.callbacks import Callbacks
from images_alignment.application.utils import add, FilesSelector, Terminal
from images_alignment import REG_MODELS

FONT = ('Helvetica', 8, 'bold')


class View(Callbacks):
    """
    Gui associated to the spectra fitting application

    Attributes
    ----------
    ax_map, canvas_map: Matplotlib.Axes, FigureCanvasTkAgg
        Axes and canvas related to the 2D-map figure displaying
    figure_settings: FigureSettings obj
        Tkinter.TopLevel derivative object for figure parameters setting
    fit_settings: FitSettings obj
        Tkinter.TopLevel derivative object for fitting parameters setting
    paramsview: ParamsView obj
        Tkinter.TopLevel derivative object for fitting models params displaying
    statsview: StatsView obj
        Tkinter.TopLevel derivative object for fitting stats results displaying
    progressbar: ProgressBar obj
        Tkinter.TopLevel derivative object with progression bar
    range_min, range_max: Tkinter.StringVars
        Range associated to the spectrum support
    outliers_coef: Tkinter.DoubleVar
        Coefficient applied to the outliers limits
    baseline_mode: Tkinter.StringVar
        Method associated with the baseline determination method ('Semi-Auto',
        'Linear' or 'Polynomial')
    baseline_coef: Tkinter.IntVar
        Smoothing coefficient used when calculating the baseline with the
        'Semi-Auto' algorithm
    baseline_attached: Tkinter.BooleanVar
        Activation keyword for baseline points attachment to the spectra
    baseline_sigma: Tkinter.IntVar
        Smoothing gaussian coefficient applied to the spectra when calculating
        the attached baseline points
    baseline_distance: Tkinter.DoubleVar
        Minimum distance used by 'spectrum.auto_baseline'
    baseline_mode: Tkinter.StringVar
        Type of baseline ('Linear' or 'Polynomial')
    baseline_order_max: Tkinter.IntVar
        Max polynomial order to consider when plotting/removing the baseline
    normalize: Tkinter.BooleanVar
        Activation keyword for spectrum profiles normalization
    normalize_range_min, normalize_range_max: Tkinter.StringVars
        Ranges for searching the maximum value used in the normalization
    model: Tkinter.StringVar
        Spectrum peak base model name among 'Gaussian', 'Lorentzian',
        'GaussianAsym' and 'LorentzianAsym'
    bkg_name: Tkinter.StringVar
        Background model name among 'None', 'Constant', 'Linear', 'Parabolic'
        and 'Exponential'
    asym: Tkinter.BooleanVar
        Activation keyword to consider asymetric spectrum model
    ax: Matplotlib.Axes object
        Current axis to work with
    canvas: FigureCanvasTkAgg object
        Current canvas to work with
    fileselector: common.core.appli_gui.FilesSelector object
        Widget dedicated to the files selection
    """

    def __init__(self, root, model):

        super().__init__()

        self.model = model
        self.areas_entry = [None, None]
        self.thresholds = [DoubleVar(value=self.model.thresholds[0]),
                           DoubleVar(value=self.model.thresholds[1])]
        self.registration_model = StringVar(value=self.model.registration_model)
        self.color = StringVar(value='Gray')
        self.mode = StringVar(value='Juxtaposed')
        self.show_results = BooleanVar(value=True)

        # Frames creation
        #################

        frame = Frame(root)

        frame_proc = Frame(frame)
        frame_proc.grid(row=0, column=0, padx=0, sticky=W + E)

        frame_visu = Frame(frame)
        frame_visu.grid(row=0, column=1, sticky=W + E)

        frame.pack()

        # VISU frame
        ############

        fig0, ax0 = plt.subplots(1, 4, figsize=(11, 1.5),
                                 gridspec_kw={'width_ratios': [1, 1, 1, 2]})
        fig0.tight_layout(h_pad=0)
        fig0.subplots_adjust(bottom=0.02)
        for i in range(4):
            ax0[i].set_label(i)
            ax0[i].get_xaxis().set_visible(False)
            ax0[i].get_yaxis().set_visible(False)
        self.model.ax = [ax0[i] for i in range(4)]

        fig1, self.ax1 = plt.subplots(1, 1, figsize=(11, 6))
        plt.tight_layout()

        frame = Frame(frame_visu)
        add(frame, 0, 0)

        fr = LabelFrame(frame)
        add(fr, 0, 0)
        add(Radiobutton(fr, text='Gray', value='Gray',
                        variable=self.color,
                        command=self.update_plots), 0, 0, pady=0)
        add(Radiobutton(fr, text='Binarized', value='Binarized',
                        variable=self.color,
                        command=self.update_plots), 0, 1, pady=0)

        self.canvas0 = FigureCanvasTkAgg(fig0, master=frame_visu)
        add(self.canvas0.get_tk_widget(), 1, 0, padx=0)
        self.canvas0.draw()
        self.canvas0.mpl_connect('button_press_event', self.select_axis)

        self.canvas1 = FigureCanvasTkAgg(fig1, master=frame_visu)
        add(self.canvas1.get_tk_widget(), 2, 0, padx=0)
        self.canvas1.draw()
        self.canvas1.mpl_connect('button_press_event', self.init_rectangle)
        self.canvas1.mpl_connect('motion_notify_event', self.draw_rectangle)
        self.canvas1.mpl_connect('button_release_event', self.set_area)
        self.canvas1.mpl_connect('button_press_event', self.init_or_remove_line)
        self.canvas1.mpl_connect('motion_notify_event', self.draw_line)
        self.canvas1.mpl_connect('scroll_event', self.zoom)

        fr_toolbar = Frame(frame_visu)
        add(fr_toolbar, 3, 0, W)
        self.toolbar = NavigationToolbar2Tk(self.canvas1, fr_toolbar)

        # PROCESSING frame
        ##################

        self.fselectors = []
        for k, label in enumerate(['Fixed image', 'Moving image']):
            frame = LabelFrame(frame_proc, text=label, font=FONT)
            add(frame, k, 0)

            fr = Frame(frame)
            add(fr, 0, 0, W + E, cspan=3)
            fselector = FilesSelector(root=fr, lbox_size=[45, 3])
            for event in ['<<ListboxSelect>>', '<<ListboxAdd>>',
                          '<<ListboxRemove>>', '<<ListboxRemoveAll>>']:
                fselector.lbox.bind(event, lambda _, k=k: self.update_file(k))
            self.fselectors.append(fselector)

            add(Label(frame, text='Cropping area:'), 1, 0, E, pady=0)
            self.areas_entry[k] = Entry(frame)
            self.areas_entry[k].bind("<Return>",
                                     lambda _, k=k: self.update_areas(k))
            add(self.areas_entry[k], 1, 1, W, pady=0)

            add(Label(frame, text='Threshold:'), 2, 0, E, pady=0)
            add(Scale(frame, resolution=0.01, to=1., orient=HORIZONTAL,
                      length=120, tickinterval=1, variable=self.thresholds[k],
                      command=lambda val, k=k: self.update_threshold(val, k)),
                2, 1, W, pady=0)
            add(Button(frame, text='Reverse',
                       command=lambda k=k: self.bin_inversion(k)), 2, 2, pady=0)

            frame = LabelFrame(frame_proc, text='Preprocessing', font=FONT)
            add(frame, 2, 0, W + E)

            for i, reg_model in enumerate(REG_MODELS):
                add(Radiobutton(frame, text=reg_model, value=reg_model,
                                variable=self.registration_model,
                                command=self.update_registration_model), 0, i)

            add(Button(frame, text='REGISTRATION',
                       command=self.registration), 1, 1)

            add(Button(frame, text='SAVE IMAGES',
                       command=self.save_images), 2, 0)
            add(Button(frame, text='SAVE MODEL',
                       command=self.model.save_model), 2, 1)
            add(Button(frame, text='LOAD MODEL',
                       command=self.reload_model), 2, 2)

            frame = LabelFrame(frame_proc, text='Application', font=FONT)
            add(frame, 3, 0, W + E)

            add(Button(frame, text='SELECT DIR. RESULT',
                       command=self.model.set_dirname_res), 0, 0, cspan=2)

            add(Checkbutton(frame, text='Fixed registration',
                            variable=self.model.fixed_reg), 1, 0, padx=30)
            add(Button(frame, text='APPLY TO ALL',
                       command=self.apply_to_all), 1, 1, padx=20)

            add(Checkbutton(frame, text='Show results',
                            variable=self.show_results,
                            command=self.plot_results), 2, 0, cspan=2)

            self.model.terminal = Terminal(frame_proc)
            add(self.model.terminal, 4, 0, W + E, cspan=3)
