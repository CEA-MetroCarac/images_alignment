import os
import platform
import warnings
import re

from tkinter import (Tk, Toplevel, Frame, LabelFrame, Label, Radiobutton, Scale,
                     Entry, Text, Button, Checkbutton, messagebox, W, E, END,
                     HORIZONTAL, IntVar, DoubleVar, StringVar, BooleanVar,
                     Listbox, EXTENDED, BOTTOM, X, Y, LEFT, RIGHT)
from tkinter.ttk import Combobox, Scrollbar
from tkinter import filedialog as fd
from tkinter.messagebox import askyesno
import itertools
from pathlib import Path
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.colors import ListedColormap

from images_alignment import ImagesAlign
from images_alignment import REG_MODELS, KEYS, AXES_NAMES, STEP, ANGLE, COEF

REG_MODELS += ['User-Driven']
VIEW_MODES = ['Gray', 'Binarized', 'Juxtaposed']
FONT = ('Helvetica', 8, 'bold')
CMAP_BINARIZED = ListedColormap(["#00FF00", "black", "red"])


def add(obj, row, col, sticky='', padx=5, pady=3, rspan=1, cspan=1, **kwargs):
    """ Add tkinter object at the (row, col)-position of a grid """
    obj.grid(row=row, column=col, sticky=sticky, padx=padx, pady=pady,
             rowspan=rspan, columnspan=cspan, **kwargs)


def hsorted(list_):
    """ Sort the given list in the way that humans expect """
    list_ = [str(x) for x in list_]
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(list_, key=alphanum_key)


class View:
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

        self.model = model

        self.thresholds = [DoubleVar(value=self.model.thresholds[0]),
                           DoubleVar(value=self.model.thresholds[1])]
        self.bin_inversions = [BooleanVar(value=self.model.bin_inversions[0]),
                               BooleanVar(value=self.model.bin_inversions[0])]
        self.registration_model = StringVar(value=self.model.registration_model)

        self.k_ref = 2
        self.pair = [None, None]
        self.line = None

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

        gs_kw = dict(width_ratios=[1, 4])
        fig, ax = plt.subplot_mosaic([[0, 3], [1, 3], [2, 3]],
                                     gridspec_kw=gs_kw,
                                     figsize=(12, 8),
                                     layout="constrained")
        self.model.ax = [ax[i] for i in range(4)]

        frame = Frame(frame_visu)
        add(frame, 0, 0, pady=0)
        self.view_mode = StringVar(value=VIEW_MODES[0])
        add(Radiobutton(frame, text=VIEW_MODES[0], value=VIEW_MODES[0],
                        variable=self.view_mode,
                        command=self.update_plots), 0, 0, W, pady=0)
        add(Radiobutton(frame, text=VIEW_MODES[1], value=VIEW_MODES[1],
                        variable=self.view_mode,
                        command=self.update_plots), 0, 1, pady=0)
        add(Radiobutton(frame, text=VIEW_MODES[2], value=VIEW_MODES[2],
                        variable=self.view_mode,
                        command=self.update_plots), 0, 2, E, pady=0)

        self.canvas = FigureCanvasTkAgg(fig, master=frame_visu)
        add(self.canvas.get_tk_widget(), 2, 0)
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        fr_toolbar = Frame(frame_visu)
        add(fr_toolbar, 3, 0, W)
        self.toolbar = NavigationToolbar2Tk(self.canvas, fr_toolbar)

        # IMG1 SETTINGS frame
        #####################

        self.fselectors = []
        for k, label in enumerate(['Fixed image', 'Moving image']):
            frame = LabelFrame(frame_proc, text=label, font=FONT)
            add(frame, k, 0)

            fr = Frame(frame)
            add(fr, 0, 0, W + E, cspan=3)
            fselector = FilesSelector(root=fr, lbox_size=[45, 3])
            fselector.lbox.bind('<<ListboxSelect>>',
                                lambda _, k=k: self.update(k))
            # fselector.lbox.bind('<<ListboxAdd>>',
            #                     lambda _, k=k: self.add_items)
            # fselector.lbox.bind('<<ListboxRemove>>', self.delete)
            # fselector.lbox.bind('<<ListboxRemoveAll>>', self.delete_all)
            self.fselectors.append(fselector)

            add(Button(frame, text='REINIT',
                       command=lambda k=k: self.reinit(k)), 1, 1)

            add(Label(frame, text='Threshold:'), 2, 0)
            add(Scale(frame, resolution=0.01, to=1., orient=HORIZONTAL,
                      tickinterval=1, variable=self.thresholds[k],
                      command=lambda v, k=k: self.update_threshold(v, k)), 2, 1)
            add(Checkbutton(frame, text='Reversed',
                            variable=self.bin_inversions[k],
                            command=lambda k=k: self.update_reversed(k)), 2, 2)

        frame = LabelFrame(frame_proc, text='Preprocessing', font=FONT)
        add(frame, 2, 0, W + E)

        add(Button(frame, text='CROPPING', command=self.cropping), 0, 0)
        add(Button(frame, text='RESIZING', command=self.resizing), 0, 1)
        add(Button(frame, text='REGISTRATION', command=self.registration), 0, 2)

        add(Radiobutton(frame, text=REG_MODELS[0], value=REG_MODELS[0],
                        variable=self.registration_model), 1, 0)
        add(Radiobutton(frame, text=REG_MODELS[1], value=REG_MODELS[1],
                        variable=self.registration_model), 1, 1)
        add(Radiobutton(frame, text=REG_MODELS[2], value=REG_MODELS[2],
                        variable=self.registration_model), 1, 2)

    def on_press(self, event):
        if self.toolbar.mode != '' and event.inaxes not in self.model.ax:
            return

        # set the clicked axis to be the ref. axis to be displayed in ax[3]
        if event.inaxes in self.model.ax[:3]:
            self.k_ref = int(event.inaxes.axes.get_label())
            self.update_plot_3()
            self.canvas.draw()

        # 'User-Driven' points selection
        elif self.view_mode.get() != 'User-Driven':
            x, y = event.xdata, event.ydata
            x12 = self.model.imgs[0].shape[1]
            if x > x12:
                if self.pair[1] is None:
                    self.pair[1] = [x, y]
            else:
                if self.pair[0] is None:
                    self.pair[0] = [x, y]
            if None not in self.pair:
                (x1, y1), (x2, y2) = self.pair
                self.model.ax[3].plot((x1, x2), (y1, y2), 'r-')
                self.model.points[0].append([x1, y1])
                self.model.points[1].append([x2 - x12, y2])
                self.canvas.draw()
                self.pair = [None, None]
                if self.line is not None:
                    self.line.remove()
                self.line = None

    def on_motion(self, event):
        if self.toolbar.mode != '' or event.inaxes != self.model.ax[3]:
            return

        if self.pair == [None, None]:
            return

        if self.line is not None:
            self.line.remove()

        x, y = event.xdata, event.ydata
        if self.pair[1] is None:
            x1, y1 = self.pair[0]
            self.line, = self.model.ax[3].plot((x1, x), (y1, y), 'r-')
        else:
            x2, y2 = self.pair[1]
            self.line, = self.model.ax[3].plot((x, x2), (y, y2), 'r-')
        self.canvas.draw()

    def update(self, k):
        ind = self.fselectors[k].lbox.curselection()[0]
        self.model.load_image(k, fname=self.fselectors[k].fnames[ind])

        if k == 0:
            self.fselectors[1].select_item(ind)
            self.model.load_image(1, fname=self.fselectors[1].fnames[ind])

        elif len(self.fselectors[0].fnames) > 1:
            self.fselectors[0].select_item(ind)
            self.model.load_image(0, fname=self.fselectors[0].fnames[ind])

        self.update_plots()

    def update_plots(self, k=None):
        view_mode = self.view_mode.get()
        if k is None:
            self.model.plot(mode=view_mode)
        elif k in range(2):
            self.model.plot_k(k, mode=view_mode)
            self.model.plot_k(2, mode=view_mode)
        self.update_plot_3()
        self.canvas.draw()

    def update_plot_3(self):

        self.model.ax[3].clear()
        ax_ref = self.model.ax[self.k_ref]

        imgs = ax_ref.get_images()
        if len(imgs) > 0:
            arr = imgs[0].get_array()
            view_mode = self.view_mode.get()
            if view_mode == 'Binarized':
                cmap, vmin, vmax = CMAP_BINARIZED, -1, 1
            else:
                cmap, vmin, vmax = 'gray', None, None
            self.model.ax[3].imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)

        lines = ax_ref.get_lines()
        for line in lines:
            color = (255 * line.get_color()).astype(int)
            hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
            self.model.ax[3].plot(line.get_xdata(), line.get_ydata(),
                                  c=hex_color, lw=2)

        if self.k_ref == 2 and self.view_mode.get() == 'Juxtaposed':
            self.model.ax[3].axvline(self.model.imgs[0].shape[1],
                                     c='w', ls='dashed', lw=0.5)

    def update_threshold(self, value, k):
        self.model.thresholds[k] = float(value)
        # self.model.binarization_k(k)
        self.model.registration_apply()
        self.update_plots(k)

    def update_reversed(self, k):
        self.model.bin_inversions[k] = self.bin_inversions[k].get()
        # self.model.binarization_k(k)
        self.model.registration_apply()
        self.update_plots(k)

    def reinit(self, k):
        """ Reinit the k-th image """
        ind = self.fselectors[k].lbox.curselection()[0]
        fname = self.fselectors[k].fnames[ind]
        self.model.load_image(k, fname=fname, reinit=True)
        self.update_plots()

    def cropping(self):
        """ Crop the images """
        k = self.k_ref
        if k not in [0, 1]:
            return
        x, y = self.model.ax[3].get_xlim(), self.model.ax[3].get_ylim()
        area = [int(y[1]), int(y[0]), int(x[0]), int(x[1])]
        self.model.cropping_areas[k] = area
        self.model.cropping(k)
        self.update_plots(k)

    def resizing(self):
        """ Resize the images """
        self.model.resizing()
        self.update_plots()

    def registration(self):
        """ Apply registration """
        registration_model = self.registration_model.get()
        self.model.registration(registration_model=registration_model)
        self.update_plots()


class FilesSelector:
    """
    Class dedicated to the files selection

    Attributes
    ----------
    lbox: list of Listbox object
        Listbox associated to the selected files
    fnames: list of str
        List of filenames to work with

    Parameters
    ----------
    root: Tk.widget
        The main window associated to the FileSelector
    lbox_size: list of 2 ints, optional
        Size (width, height) of the Listbox 'lbox'. Default value is [30, 15]
    """

    def __init__(self, root, lbox_size=None):

        lbox_size = lbox_size or [30, 15]

        self.fnames = []

        # create buttons and listbox

        Button(root, text="Select Files",
               command=self.select_files). \
            grid(column=0, row=0, padx=0, pady=5)
        Button(root, text="Select Dir.",
               command=self.select_dir). \
            grid(column=1, row=0, padx=0, pady=5)
        Button(root, text="Remove",
               command=self.remove). \
            grid(column=2, row=0, padx=0, pady=5)
        Button(root, text="Remove all",
               command=self.remove_all). \
            grid(column=3, row=0, padx=0, pady=5)

        Button(root, text="▲", command=lambda: self.move('up')). \
            grid(column=1, row=1, pady=0, sticky=E)
        Button(root, text="▼", command=lambda: self.move('down')). \
            grid(column=2, row=1, pady=0, sticky=W)

        lbox_frame = Frame(root)
        lbox_frame.grid(column=0, row=2, padx=5, pady=0, columnspan=4)
        sbar_v = Scrollbar(lbox_frame)
        # sbar_h = Scrollbar(lbox_frame, orient=HORIZONTAL)
        self.lbox = Listbox(lbox_frame,
                            width=lbox_size[0],
                            height=lbox_size[1],
                            # selectmode=EXTENDED,
                            activestyle="underline",
                            exportselection=False,
                            # xscrollcommand=sbar_h.set,
                            yscrollcommand=sbar_v.set)

        sbar_v.config(command=self.lbox.yview)
        # sbar_h.config(command=self.lbox.xview)

        # sbar_h.pack(side=BOTTOM, fill=X)
        self.lbox.pack(side=LEFT, fill=Y)
        sbar_v.pack(side=RIGHT, fill=Y)

    def move(self, key):
        """ Move cursor selection according to key value (up or down) """
        increment = {'up': -1, 'down': 1, 'none': 0}
        indices = self.lbox.curselection()
        if not indices:
            return
        ind = min(indices)  # working with several indices has no sense
        if len(indices) > 1:
            key = 'none'
        elif (key == 'up' and ind == 0) or \
                (key == 'down' and ind == len(self.fnames) - 1):
            return
        ind += increment[key]
        self.select_item(ind)

        self.lbox.event_generate('<<ListboxSelect>>')

    def add_items(self, fnames=None, ind_start=None):
        """ Add items from a 'fnames' list """
        if fnames is None:
            return

        ind_start = ind_start or len(self.fnames)

        for fname in hsorted(fnames):
            self.lbox.insert(END, os.path.basename(fname))
            self.fnames.append(fname)

        # select the first new item
        self.select_item(ind_start)

    def select_files(self, fnames=None):
        """ Add items from selected files """

        if fnames is None:
            fnames = fd.askopenfilenames(title='Select file(s)')
        self.add_items(fnames=fnames)

        self.lbox.event_generate('<<ListboxAdd>>')

    def select_dir(self, dirname=None):
        """ Add items from a directory """
        if dirname is None:
            dirname = fd.askdirectory(title='Select directory')

        ind_start = len(self.fnames)
        fnames = glob.glob(os.path.join(dirname, '*.txt'))
        self.add_items(fnames, ind_start=ind_start)

        self.lbox.event_generate('<<ListboxAdd>>')

    def remove(self):
        """ Remove selected items """
        selected_items = self.lbox.curselection()

        # deleting one by one item is too long when working with a big selection
        groups = groupby(selected_items, key=lambda n, c=count(): n - next(c))
        groups = [list(g) for _, g in groups]
        for group in groups:
            self.lbox.delete(group[0], group[-1])

        for selected_item in reversed(selected_items):
            self.fnames.pop(selected_item)

        # reselect the first item
        self.select_item(0)

        self.lbox.event_generate('<<ListboxRemove>>')

    def remove_all(self):
        """ Remove all items """
        self.lbox.delete(0, END)
        self.fnames = []

        self.lbox.event_generate('<<ListboxRemoveAll>>')

    def select_item(self, index, selection_clear=True):
        """ Select item in the listbox """
        if selection_clear:
            self.lbox.selection_clear(0, END)
        self.lbox.selection_set(index)
        self.lbox.activate(index)
        self.lbox.selection_anchor(index)
        self.lbox.see(index)


class App:
    """
    Application for spectra fitting

    Attributes
    ----------
    root: Tkinter.Tk object
        Root window
    force_terminal_exit: bool
        Key to force terminal session to exit after 'root' destroying
    """

    def __init__(self,
                 root, size="1550x950", force_terminal_exit=True,
                 fnames_fixed=None,
                 fnames_moving=None,
                 thresholds=None,
                 bin_inversions=None
                 ):
        root.title("images_alignment")
        root.geometry(size)
        self.root = root
        self.force_terminal_exit = force_terminal_exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.model = ImagesAlign(fnames_fixed=fnames_fixed,
                                 fnames_moving=fnames_moving,
                                 thresholds=thresholds,
                                 bin_inversions=bin_inversions)

        self.view = View(self.root, self.model)
        self.view.fselectors[0].add_items(fnames=fnames_fixed)
        self.view.fselectors[1].add_items(fnames=fnames_moving)

    def on_closing(self):
        """ To quit 'properly' the application """
        if messagebox.askokcancel("Quit", "Would you like to quit ?"):
            self.root.destroy()


def launcher(fname_json=None):
    """ Launch the appli """

    root = Tk()
    appli = App(root)

    if fname_json is not None:
        appli.reload(fname_json=fname_json)

    # dirname = Path(r"C:\Users\PQ177701\AppData\Local\Temp\images_alignement")
    # fnames = [dirname / "img2_1.tif",
    #           dirname / "img2_2.tif",
    #           dirname / "img2_3.tif"]
    #
    # appli.fselectors[0].add_items(fnames=[dirname / "img2_2.tif"])
    # appli.fselectors[1].add_items(fnames=fnames)

    root.mainloop()


if __name__ == '__main__':
    launcher()
