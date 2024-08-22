"""
Class Callbacks attached to the application
"""
from pathlib import Path
from tkinter import END
from tkinter import filedialog as fd
from tkinter.messagebox import showerror

from matplotlib.patches import Rectangle
from imageio.v3 import imwrite

from images_alignment import CMAP_BINARIZED
from images_alignment.utils import recast


class Callbacks:
    """
    Class Callbacks attached to the application
    """

    def __init__(self):

        self.k_ref = 2
        self.pair = [None, None]
        self.rectangle = None
        self.line = None
        self.lines = []

        self.areas = [None, None]
        self.fnames_tot = [None, None]

    def select_axis(self, event):
        """ Select the axis to be displayed in 'fig1' """
        if self.toolbar.mode != '' or event.inaxes not in self.model.ax:
            return

        self.k_ref = int(event.inaxes.axes.get_label())
        self.update_fig1()
        self.canvas1.draw()

    def init_rectangle(self, event):
        """ Initialize rectangle """
        if self.toolbar.mode != '' or event.inaxes != self.ax1:
            return

        if self.k_ref not in [0, 1]:
            return

        x, y = event.xdata, event.ydata
        self.pair = [[x, y], [None, None]]
        self.rectangle = Rectangle((x, y), 0, 0, ec='y', fc='none')
        self.ax1.add_patch(self.rectangle)

    def init_or_remove_line(self, event):
        """ Initialize or Remove a line """
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
        """ Draw a rectangle """

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
            shape = self.model.imgs[self.k_ref].shape
            area = [max(0, int(min(x0, x))), min(shape[1], int(max(x0, x))),
                    max(0, int(min(y0, y))), min(shape[0], int(max(y0, y)))]
            self.model.set_area_k(self.k_ref, area=area)
            self.update_plots(self.k_ref)
            self.pair = [None, None]
            self.areas_entry[self.k_ref].delete(0, END)
            self.areas_entry[self.k_ref].insert(0, str(area))

    def set_area(self, event):
        """ Set area from the view to the model """
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

    def update(self):
        """ Update the view parameters from the model and the plots """
        for k in range(2):
            self.areas_entry[k].delete(0, END)
            self.areas_entry[k].insert(0, str(self.model.areas[k]))
        self.registration_model.set(self.model.registration_model)
        self.update_plots()

    def update_file(self, k):
        """ Update the k-th image from the fileselector and its related one """
        fsel = self.fselectors

        ind = fsel[k].lbox.curselection()[0]
        self.model.load_image(k, fname=fsel[k].fnames[ind])

        if k == 0:
            fsel[1].select_item(ind)
            self.model.load_image(1, fname=fsel[1].fnames[ind])

        elif len(self.fselectors[0].fnames) > 1:
            fsel[0].select_item(ind)
            self.model.load_image(0, fname=fsel[0].fnames[ind])

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
        self.update_fig1()
        self.canvas0.draw()
        self.canvas1.draw()

    def update_fig1(self):
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

    def update_areas(self, k):
        """ Update areas of the k-th image from the Tkinter.Entry """
        area_entry = self.areas_entry[k].get()
        if area_entry == '':
            self.model.areas[k] = None
        else:
            try:
                self.model.areas[k] = eval(area_entry)
            except:
                msg = f"{self.areas_entry[k].get()}  cannot be interpreted"
                msg += " as '[xmin, xmax, ymin, ymax]'"
                showerror(message=msg)
                return
        self.update_plots(k)

    def update_threshold(self, value, k):
        """ Update the threshold value associated with the k-th image """
        self.model.thresholds[k] = float(value)
        self.model.registration_apply()
        self.update_plots(k)

    def update_registration_model(self):
        """ Update the threshold value associated with the k-th image """
        self.model.registration_model = self.registration_model.get()
        self.model.reinit()
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
        """ Save the fixed and moving images in their final state """
        imgs = self.model.crop_and_resize(self.model.imgs)
        if self.model.img_reg is not None:
            imgs[1] = self.model.img_reg

        for k in range(2):
            ind = self.fselectors[k].lbox.curselection()[0]
            fname = Path(self.fselectors[k].fnames[ind])
            initialdir = fname.parent
            initialfile = fname.stem + "_aligned" + fname.suffix
            fname_reg = fd.asksaveasfilename(initialfile=initialfile,
                                             initialdir=initialdir)
            if fname_reg != "":
                imwrite(fname_reg, recast(imgs[k], self.model.dtypes[k]))

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
        self.model.apply_to_all(dirname_res=dirname_res)
        self.areas = self.model.areas
        self.fnames_tot = self.model.fnames_tot
        self.plot_results()

    def plot_results(self):
        """ Display the figures related to the 'dirname_res' """
        if self.model.dirname_res == [None, None]:
            return

        if self.show_results.get():
            model = self.model
            for k in range(2):
                self.model.areas = [None, None]
                self.fselectors[k].fnames = [model.dirname_res[k] / Path(x).name
                                             for x in self.fnames_tot[k]]
        else:
            for k in range(2):
                self.model.areas = self.areas
                self.fselectors[k].fnames = self.fnames_tot[k]

        self.update_file(0)
