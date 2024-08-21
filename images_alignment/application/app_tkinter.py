"""
Tkinter-application for images alignment
"""
import os
import re
import glob
from pathlib import Path
from itertools import groupby, count
from tkinter import (Tk, Frame, LabelFrame, Label, Radiobutton, Scale,
                     Button, Checkbutton, Listbox, Entry, messagebox,
                     W, E, END, HORIZONTAL, Y, LEFT, RIGHT,
                     DoubleVar, StringVar)
from tkinter.ttk import Scrollbar
from tkinter import filedialog as fd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from imageio.v3 import imwrite

from images_alignment import ImagesAlign
from images_alignment import REG_MODELS

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
        self.areas_entry = [None, None]
        self.thresholds = [DoubleVar(value=self.model.thresholds[0]),
                           DoubleVar(value=self.model.thresholds[1])]
        self.registration_model = StringVar(value=self.model.registration_model)

        self.color = StringVar(value='Gray')
        self.mode = StringVar(value='Juxtaposed')
        self.k_ref = 2
        self.pair = [None, None]
        self.rectangle = None
        self.line = None
        self.lines = []

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

        fig0, ax0 = plt.subplots(1, 4, figsize=(10, 1.8),
                                 gridspec_kw={'width_ratios': [1, 1, 1, 2]})
        fig0.tight_layout(h_pad=0.1)
        [ax0[i].set_label(i) for i in range(4)]
        [ax0[i].get_xaxis().set_visible(False) for i in range(4)]
        [ax0[i].get_yaxis().set_visible(False) for i in range(4)]
        self.model.ax = [ax0[i] for i in range(4)]

        fig1, self.ax1 = plt.subplots(1, 1, figsize=(10, 5))
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
            fselector.lbox.bind('<<ListboxSelect>>',
                                lambda _, k=k: self.update(k))
            # fselector.lbox.bind('<<ListboxAdd>>',
            #                     lambda _, k=k: self.add_items)
            # fselector.lbox.bind('<<ListboxRemove>>', self.delete)
            # fselector.lbox.bind('<<ListboxRemoveAll>>', self.delete_all)
            self.fselectors.append(fselector)

            add(Label(frame, text='Cropping area:'), 1, 0, E)
            self.areas_entry[k] = Entry(frame)
            self.areas_entry[k].insert(0, str(self.model.areas[k]))
            add(self.areas_entry[k], 1, 1, W)
            add(Button(frame, text='REINIT',
                       command=lambda k=k: self.reinit(k)), 1, 2)

            add(Label(frame, text='Threshold:'), 2, 0, E)
            add(Scale(frame, resolution=0.01, to=1., orient=HORIZONTAL,
                      tickinterval=1, variable=self.thresholds[k],
                      command=lambda val, k=k: self.update_threshold(val, k)),
                2, 1, W)
            add(Button(frame, text='Reverse',
                       command=lambda k=k: self.bin_inversion(k)), 2, 2)

        frame = LabelFrame(frame_proc, text='Preprocessing', font=FONT)
        add(frame, 2, 0, W + E)

        add(Radiobutton(frame, text=REG_MODELS[0], value=REG_MODELS[0],
                        variable=self.registration_model), 0, 0)
        add(Radiobutton(frame, text=REG_MODELS[1], value=REG_MODELS[1],
                        variable=self.registration_model), 0, 1)
        add(Radiobutton(frame, text=REG_MODELS[2], value=REG_MODELS[2],
                        variable=self.registration_model), 0, 2)

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

    def select_axis(self, event):
        """ Select the axis to be displayed in 'fig1' """
        if self.toolbar.mode != '' or event.inaxes not in self.model.ax:
            return

        self.k_ref = int(event.inaxes.axes.get_label())
        self.update_plot_1()
        self.canvas1.draw()

    def init_rectangle(self, event):
        if self.toolbar.mode != '' or event.inaxes != self.ax1:
            return

        if self.k_ref not in [0, 1]:
            return

        x, y = event.xdata, event.ydata
        self.pair = [[x, y], [None, None]]
        self.rectangle = Rectangle((x, y), 0, 0, ec='y', fc='none')
        self.ax1.add_patch(self.rectangle)

    def init_or_remove_line(self, event):
        if self.toolbar.mode != '' or event.inaxes != self.ax1:
            return

        if self.k_ref != 3 or self.registration_model.get() != 'User-Driven':
            return

        x, y = event.xdata, event.ydata
        x12 = self.model.imgs[0].shape[1]

        if event.button == 1:
            if x > x12:
                if self.pair[1] is None:
                    self.pair[1] = [x, y]
            else:
                if self.pair[0] is None:
                    self.pair[0] = [x, y]
            if None not in self.pair:
                (x1, y1), (x2, y2) = self.pair
                self.lines.append(self.ax1.plot((x1, x2), (y1, y2), 'r-')[0])
                self.model.points[0].append([x1, y1])
                self.model.points[1].append([x2 - x12, y2])
                self.remove_moving_line()

        elif event.button == 3:
            if self.pair != [None, None]:  # remove the current line
                self.remove_moving_line()
            elif len(self.lines) > 0:  # remove the closest line
                pts = self.model.points
                d0 = [(xp - x) ** 2 + (yp - y) ** 2 for xp, yp in pts[0]]
                d1 = [(xp + x12 - x) ** 2 + (yp - y) ** 2 for xp, yp in pts[1]]
                d0_min, d1_min = min(d0), min(d1)
                ind = d0.index(d0_min) if d0_min < d1_min else d1.index(d1_min)
                self.lines[ind].remove()
                del self.lines[ind]
                del self.model.points[0][ind]
                del self.model.points[1][ind]
                self.canvas1.draw_idle()

    def remove_moving_line(self):
        """ Remove the moving line """
        self.pair = [None, None]
        if self.line is not None:
            self.line.remove()
        self.line = None
        self.canvas1.draw_idle()

    def draw_line(self, event):
        """ Draw the line in 'fig1' (with 'User-Driven' mode activated) """

        if self.toolbar.mode != '' or event.inaxes != self.ax1:
            return

        if self.k_ref != 3 or self.pair == [None, None]:
            return

        if self.line is not None:
            self.line.remove()

        x, y = event.xdata, event.ydata
        if self.pair[1] is None:
            x1, y1 = self.pair[0]
            self.line, = self.ax1.plot((x1, x), (y1, y), 'r-')
        else:
            x2, y2 = self.pair[1]
            self.line, = self.ax1.plot((x, x2), (y, y2), 'r-')
        self.canvas1.draw_idle()

    def draw_rectangle(self, event, set_area=False):

        if self.toolbar.mode != '' or event.inaxes != self.ax1:
            return

        if self.k_ref not in [0, 1] or self.pair[0] is None:
            return

        x, y = event.xdata, event.ydata
        x0, y0 = self.pair[0]
        self.rectangle.set_width(x - x0)
        self.rectangle.set_height(y - y0)
        self.canvas1.draw_idle()

        if set_area:
            area = [int(min(x0, x)), int(max(x0, x)),
                    int(min(y0, y)), int(max(y0, y))]
            self.model.set_area_k(self.k_ref, area=area)
            self.areas_entry[self.k_ref].delete(0, END)
            self.areas_entry[self.k_ref].insert(0, str(area))
            self.update_plots(self.k_ref)
            self.pair = [None, None]

    def set_area(self, event):
        self.draw_rectangle(event, set_area=True)

    def zoom(self, event):
        """ Zoom/Unzoom the 'fig1' """
        base_scale = 1.1

        if event.button == 'up':
            scale_factor = base_scale
        elif event.button == 'down':
            scale_factor = 1 / base_scale
        else:
            return

        x, y = event.xdata, event.ydata

        xlim0 = self.ax1.get_xlim()
        ylim0 = self.ax1.get_ylim()

        new_width = (xlim0[1] - xlim0[0]) * scale_factor
        new_height = (ylim0[1] - ylim0[0]) * scale_factor

        relx = (xlim0[1] - x) / (xlim0[1] - xlim0[0])
        rely = (ylim0[1] - y) / (ylim0[1] - ylim0[0])

        self.ax1.set_xlim([x - new_width * (1 - relx), x + new_width * relx])
        self.ax1.set_ylim([y - new_height * (1 - rely), y + new_height * rely])

        self.ax1.figure.canvas.toolbar.push_current()
        self.canvas1.draw_idle()

    def update(self, k):
        """ Update the k-th image from the fileselector and its related one """
        fsel = self.fselectors

        ind = fsel[k].lbox.curselection()[0]
        self.model.load_image(k, fname=fsel[k].fnames[ind], reinit=True)

        if k == 0:
            fsel[1].select_item(ind)
            self.model.load_image(1, fname=fsel[1].fnames[ind], reinit=True)

        elif len(self.fselectors[0].fnames) > 1:
            fsel[0].select_item(ind)
            self.model.load_image(0, fname=fsel[0].fnames[ind], reinit=True)

        self.update_plots()

    def update_plots(self, k=None):
        """ Update the plots """
        self.model.color = self.color.get()
        self.model.mode = self.mode.get()
        if k is None:
            self.model.plot_all()
        elif k in range(2):
            self.model.plot_k(k)
            self.model.plot_k(2)
            self.model.plot_k(3)
        self.update_plot_1()
        self.canvas0.draw()
        self.canvas1.draw()

    def update_plot_1(self):
        """ Update the fig1 """

        self.ax1.clear()
        ax_ref = self.model.ax[self.k_ref]

        imgs = ax_ref.get_images()
        if len(imgs) > 0:
            arr = imgs[0].get_array()
            if self.color.get() == 'Binarized':
                cmap, vmin, vmax = CMAP_BINARIZED, -1, 1
            else:
                cmap, vmin, vmax = 'gray', None, None
            self.ax1.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)

        lines = ax_ref.get_lines()
        for line in lines:
            color = (255 * line.get_color()).astype(int)
            hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
            self.ax1.plot(line.get_xdata(), line.get_ydata(), c=hex_color, lw=2)

        if self.k_ref == 3:
            self.ax1.axvline(self.model.imgs[0].shape[1],
                             c='w', ls='dashed', lw=0.5)

    def update_threshold(self, value, k):
        """ Update the threshold value associated with the k-th image """
        self.model.thresholds[k] = float(value)
        self.model.registration_apply()
        self.update_plots(k)

    def bin_inversion(self, k):
        """ Invert the binarized k-th image """
        self.model.bin_inversions[k] = not self.model.bin_inversions[k]
        if self.model.imgs_bin[k] is not None:
            self.model.imgs_bin[k] = ~self.model.imgs_bin[k]
        self.update_plots(k)

    def reinit(self, k):
        """ Reinit the k-th image """
        ind = self.fselectors[k].lbox.curselection()[0]
        fname = self.fselectors[k].fnames[ind]
        self.model.load_image(k, fname=fname, reinit=True)
        self.areas_entry[k].delete(0, END)
        self.update_plots(k)

    def registration(self):
        """ Apply registration """
        registration_model = self.registration_model.get()
        self.model.registration(registration_model=registration_model)
        self.update_plots()

    def save_images(self):
        """ Save all the images """
        self.save_image_k(0)
        self.save_image_k(1)

    def save_image_k(self, k):
        """ Save the k-th image """
        ind = self.fselectors[k].lbox.curselection()[0]
        fname = Path(self.fselectors[k].fnames[ind])
        initialdir = fname.parent
        initialfile = fname.stem + "_aligned" + fname.suffix

        fname_reg = fd.asksaveasfilename(initialfile=initialfile,
                                         initialdir=initialdir)

        if fname_reg == "":
            return

        # TODO revisit
        img = self.model.imgs[k]
        if k == 1 and self.model.img_reg is not None:
            img = self.model.img_reg

        imwrite(fname_reg, img)

    def reload_model(self):
        """ Reload model """
        fname_json = r"C:\Users\PQ177701\Desktop\model.json"
        self.model.reload_model(fname_json, obj=self.model)
        self.registration_model.set(self.model.registration_model)
        for k in range(2):
            self.thresholds[k].set(self.model.thresholds[k])
            self.areas_entry[k].delete(0, END)
            self.areas_entry[k].insert(0, str(self.model.areas[k]))
        self.update_plots()

    def apply_to_all(self, dirname_res=None):
        """ Apply the alignment processing to all the images """
        model = self.model
        model.apply_to_all(dirname_res=dirname_res)
        for k in range(2):
            self.fselectors[k].fnames = [model.dirname_res[k] / Path(x).name
                                         for x in model.fnames_tot[k]]
        self.update(0)


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
        fnames = glob.glob(os.path.join(dirname, '*.tif'))
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
                 root, size="1350x800", force_terminal_exit=True,
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

    root.mainloop()


if __name__ == '__main__':
    launcher()