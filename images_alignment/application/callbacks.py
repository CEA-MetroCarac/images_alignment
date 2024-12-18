"""
Class Callbacks attached to the application
"""
from pathlib import Path
from tkinter import END
from tkinter import filedialog as fd
from tkinter.messagebox import showerror, askyesno

import numpy as np
from matplotlib.patches import Rectangle

from images_alignment import CMAP_BINARIZED


class Callbacks:
    """
    Class Callbacks attached to the application
    """

    def __init__(self):

        self.k_ref = 3
        self.pair = [None, None]
        self.rectangle = None
        self.line = None
        self.lines = []

        self.rois = [None, None]
        self.fnames_tot = [None, None]

    def select_axis(self, event):
        """ Select the axis to be displayed in 'fig1' """
        if event.inaxes in self.model.ax:
            k = int(event.inaxes.axes.get_label())
            if k != self.k_ref:
                self.pair = [None, None]
            self.k_ref = k
            self.update_fig1()
            self.canvas0.draw()
            self.canvas1.draw()

    def spine_axis(self):
        """ Spine the selected axis """
        for k, ax in enumerate(self.model.ax):
            color = 'red' if k == self.k_ref else 'white'
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

    def init_rectangle(self, event):
        """ Initialize rectangle """
        if self.toolbar.mode == '' and \
                event.inaxes == self.ax1 and \
                self.k_ref in [0, 1] and \
                self.pair[0] is None:
            x, y = event.xdata, event.ydata
            self.pair = [[x, y], [None, None]]
            self.rectangle = Rectangle((x, y), 0, 0, ec='y', fc='none')
            self.ax1.add_patch(self.rectangle)

    def init_or_remove_line(self, event):
        """ Initialize or Remove a line """

        if self.toolbar.mode == '' and \
                event.inaxes == self.ax1 and \
                self.k_ref == 2 and \
                self.registration_model.get() == 'User-Driven':

            x, y = event.xdata, event.ydata
            rfacs = self.model.rfactors_plotting
            alignment = self.model.juxt_alignment
            shape0, shape1 = self.model.get_shapes()
            x12 = shape0[1] * rfacs[0]
            y12 = shape0[0] * rfacs[0]

            if event.button == 1:
                if (alignment == 'horizontal' and x > x12) or \
                        (alignment == 'vertical' and y > y12):
                    if self.pair[1] is None:
                        self.pair[1] = [x, y]
                else:
                    if self.pair[0] is None:
                        self.pair[0] = [x, y]
                if None not in self.pair:
                    (x1, y1), (x2, y2) = self.pair
                    line = self.ax1.plot((x1, x2), (y1, y2), 'r-')[0]
                    self.lines.append(line)
                    x1p, y1p = x1 / rfacs[0], shape0[0] - y1 / rfacs[0]
                    if alignment == 'horizontal':
                        x2p = (x2 - x12) / rfacs[1]
                        y2p = shape1[0] - y2 / rfacs[1]
                    else:
                        x2p = x2 / rfacs[1]
                        y2p = shape1[0] - (y2 - y12) / rfacs[1]
                    self.model.points[0].append([x1p, y1p])
                    self.model.points[1].append([x2p, y2p])
                    self.remove_moving_line()

            elif event.button == 3:
                if self.pair != [None, None]:  # remove the current line
                    self.remove_moving_line()
                elif len(self.lines) > 0:  # remove the closest line
                    pts = self.model.points
                    x1, y1 = x, y
                    x2, y2 = x - x12, y
                    d1 = [(xp - x1) ** 2 + (yp - y1) ** 2 for xp, yp in pts[0]]
                    d2 = [(xp + x2) ** 2 + (yp - y2) ** 2 for xp, yp in pts[1]]
                    d1min, d2min = min(d1), min(d2)
                    ind = d1.index(d1min) if d1min < d2min else d2.index(d2min)
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
        if self.toolbar.mode == '' and \
                event.inaxes == self.ax1 and \
                self.k_ref == 2 and \
                self.pair != [None, None] and \
                self.registration_model.get() == 'User-Driven':

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

    def draw_rectangle(self, event, set_roi=False):
        """ Draw a rectangle """
        if self.toolbar.mode == '' and \
                event.inaxes == self.ax1 and \
                self.k_ref in [0, 1] and \
                self.pair[0] is not None:

            x, y = event.xdata, event.ydata
            x0, y0 = self.pair[0]
            self.rectangle.set_width(x - x0)
            self.rectangle.set_height(y - y0)
            self.canvas1.draw_idle()

            if event.button == 3:
                self.pair = [None, None]
                self.rectangle.remove()
                self.rectangle = None
                self.rois_entry[self.k_ref].delete(0, END)
                self.update_rois(self.k_ref)

            if set_roi:
                shape = self.model.imgs[self.k_ref].shape
                roi = [max(0, int(min(x0, x))), min(shape[1], int(max(x0, x))),
                       max(0, int(min(y0, y))), min(shape[0], int(max(y0, y)))]
                if roi[0] != roi[1] and roi[2] != roi[3]:
                    self.rois_entry[self.k_ref].delete(0, END)
                    self.rois_entry[self.k_ref].insert(0, str(roi))
                    self.update_rois(self.k_ref)
                self.pair = [None, None]

    def set_roi(self, event):
        """ Set ROI from the view to the model """
        self.draw_rectangle(event, set_roi=True)

    def zoom(self, event):
        """ Zoom/Unzoom the 'fig1' """
        if event.inaxes != self.ax1:
            return

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

    def update(self):
        """ Update the view parameters from the model and the plots """
        for k in range(2):
            self.rois_entry[k].delete(0, END)
            self.rois_entry[k].insert(0, str(self.model.rois[k]))
        self.registration_model.set(self.model.registration_model)
        self.update_plots()

    def update_file(self, k):
        """ Update the k-th image from the fileselector and its related one """
        fsel = self.fselectors

        # synchronize fnames_tot
        self.model.fnames_tot = [fsel[0].fnames, fsel[1].fnames]

        if len(fsel[k].fnames) == 0:
            self.model.fnames[k] = None
            self.model.imgs[k] = None
            self.clear_plots()
            return

        ind = fsel[k].lbox.curselection()[0]
        fname = fsel[k].fnames[ind]
        if fname != self.model.fnames[k]:
            self.model.load_image(k, fname=fname)

        if len(fsel[0].fnames) == len(fsel[1].fnames):
            fsel[1 - k].select_item(ind)
            fname = fsel[1 - k].fnames[ind]
            if fname != self.model.fnames[1 - k]:
                self.model.load_image(1 - k, fname=fname)

        elif len(fsel[0].fnames) == 1:
            fsel[0].select_item(0)
            fname = fsel[0].fnames[0]
            if fname != self.model.fnames[0]:
                self.model.load_image(0, fname=fname)

        else:
            self.model.fnames[1 - k] = None
            self.model.imgs[1 - k] = None
            self.clear_plots()
            return

        self.update_plots()

    def update_plots(self, k=None):
        """ Update the plots """
        self.model.update_rfactors_plotting()
        self.model.binarized = self.binarized.get()
        # self.model.mode = self.mode.get()
        if k is None:
            self.model.plot_all()
        else:
            self.model.plot_k(k)
        if k in range(2):
            self.model.plot_k(3)
        self.update_fig1()
        self.canvas0.draw()
        self.canvas1.draw()

    def clear_plots(self):
        """ Clear points """
        [self.model.ax[k].clear() for k in range(4)]
        self.ax1.clear()
        self.canvas0.draw()
        self.canvas1.draw()

    def update_fig1(self):
        """ Update the fig1 """

        self.ax1.clear()
        k_ref = self.k_ref
        ax_ref = self.model.ax[k_ref]
        self.spine_axis()

        if self.binarized.get():
            cmap, vmin, vmax = CMAP_BINARIZED, -1, 1
        else:
            cmap, vmin, vmax = 'gray', None, None

        imgs = ax_ref.get_images()
        if len(imgs) > 0:
            arr = imgs[0].get_array()
            shape = arr.shape
            if k_ref in [0, 1]:
                shape = self.model.imgs[k_ref].shape
            extent = [0, shape[1], 0, shape[0]]
            self.ax1.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)

        lines = ax_ref.get_lines()
        for line in lines:
            color = (255 * line.get_color()).astype(int)
            hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
            self.ax1.plot(line.get_xdata(), line.get_ydata(), c=hex_color, lw=2)

        if k_ref in [0, 1]:
            for rect in ax_ref.patches:
                rect2 = Rectangle(np.asarray(rect.get_xy()), rect.get_width(),
                                  rect.get_height(), ec='y', fc='none')
                self.ax1.add_patch(rect2)

        if k_ref == 2:
            rfac = self.model.rfactors_plotting[0]
            shape = self.model.get_shapes()[0]
            if self.model.juxt_alignment == 'horizontal':
                self.ax1.axvline(shape[1] * rfac, c='w', ls='dashed', lw=0.5)
            elif self.model.juxt_alignment == 'vertical':
                self.ax1.axhline(shape[0] * rfac, c='w', ls='dashed', lw=0.5)

    def update_rois(self, k):
        """ Update ROIs of the k-th image from the Tkinter.Entry """
        roi_entry = self.rois_entry[k].get()
        if roi_entry == '':
            self.model.rois[k] = None
        else:
            try:
                self.model.rois[k] = eval(roi_entry)
            except:
                msg = f"{self.rois_entry[k].get()}  cannot be interpreted"
                msg += " as '[xmin, xmax, ymin, ymax]'"
                showerror(message=msg)
                return
        self.model.binarization_k(k)
        self.model.points = [[], []]
        self.model.img_reg = None
        self.model.img_reg_bin = None
        self.model.mask = None
        self.update_plots()

    def update_threshold(self, value, k):
        """ Update the threshold value associated with the k-th image """
        self.model.thresholds[k] = float(value)
        self.model.binarization_k(k)
        # self.model.registration()
        self.update_plots()

    def update_registration_model(self):
        """ Update the threshold value """
        self.model.registration_model = self.registration_model.get()
        self.model.reinit()
        self.update_plots()

    def update_inv_reg(self):
        """ Update the 'inv_reg' value """
        self.model.inv_reg = self.inv_reg.get()
        self.model.registration_apply()
        self.update_plots(k=3)

    def update_fixed_reg(self):
        """ Update the 'fixed_reg' value """
        self.model.fixed_reg = self.fixed_reg.get()

    def update_juxt_alignment(self):
        """ Update the 'juxt_alignment' value """
        self.model.juxt_alignment = self.juxt_alignment.get()
        self.update_plots(k=2)

    def update_apply_mask(self):
        """ Update the 'apply_mask' value """
        self.model.apply_mask = self.apply_mask.get()
        self.update_plots(k=3)

    def update_angles(self, k):
        """ Update the angle of k-th image """
        self.model.angles[k] = self.angles[k].get()
        self.model.load_image(k, fname=self.model.fnames[k])
        self.update_plots()

    def update_resolution(self):
        """ Update the 'resolution' and 'rfactors_plotting' values """
        self.model.resolution = self.resolution.get()
        self.model.min_img_res = int(self.min_img_res.get())
        self.model.update_rfactors_plotting()
        self.update_plots()

    def bin_inversion(self, k):
        """ Invert the binarized k-th image """
        self.model.bin_inversions[k] = not self.model.bin_inversions[k]
        if self.model.imgs_bin[k] is not None:
            self.model.imgs_bin[k] = ~self.model.imgs_bin[k]
        self.update_plots(k)

    def registration(self):
        """ Apply registration """
        registration_model = self.registration_model.get()
        self.model.registration(registration_model=registration_model)
        self.update_plots()

    def save_images(self):
        """ Save the fixed and moving images in their final states """
        fnames_save = []
        for k in range(2):
            ind = self.fselectors[k].lbox.curselection()[0]
            fname = Path(self.fselectors[k].fnames[ind])
            initialdir = fname.parent
            initialfile = fname.stem + "_aligned" + fname.suffix
            fnames_save.append(fd.asksaveasfilename(initialfile=initialfile, initialdir=initialdir))
        self.model.save_images(fnames_save)

    def reload_params(self):
        """ Reload parameters """
        self.model.reload_params(obj=self.model)
        self.registration_model.set(self.model.registration_model)
        for k in range(2):
            self.thresholds[k].set(self.model.thresholds[k])
            self.rois_entry[k].delete(0, END)
            self.rois_entry[k].insert(0, str(self.model.rois[k]))
        self.update_plots()

    def apply_to_all(self, dirname_res=None):
        """ Apply the alignment processing to all the images """
        dirnames_res = [dirname_res]
        if self.model.dirname_res[0] is not None:
            dirnames_res.append(Path(self.model.dirname_res[0]).parent)
        if Path(self.model.fnames[0]).parents[1] in dirnames_res:
            msg = 'You are about to re-align images located in the RESULT dir.'
            if self.show_results.get():
                msg += '\nTo go back to the raw data files, select "No" ' \
                       'and uncheck the "Show results" button.'
            msg += '\nContinue ?'
            if not askyesno(message=msg):
                return

        self.model.apply_to_all(dirname_res=dirname_res)
        self.rois = self.model.rois
        self.fnames_tot = self.model.fnames_tot
        self.plot_results()

    def plot_results(self):
        """ Display the figures related to the 'dirname_res' """
        if self.model.dirname_res == [None, None]:
            return

        if self.show_results.get():
            model = self.model
            for k in range(2):
                self.model.rois = [None, None]
                self.fselectors[k].fnames = [model.dirname_res[k] / Path(x).name
                                             for x in self.fnames_tot[k]]
        else:
            for k in range(2):
                self.model.rois = self.rois
                self.fselectors[k].fnames = self.fnames_tot[k]

        self.model.fnames = [None, None]  # To force the updating
        self.update_file(1)
