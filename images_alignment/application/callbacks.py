"""
Class Callbacks attached with the application
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QMessageBox, QFileDialog as fd

from pathlib import Path
import numpy as np
from matplotlib.patches import Rectangle

from images_alignment import CMAP_BINARIZED, REG_MODELS, REG_KEYS, WARP_ORDERS, COLORS


class Callbacks:
    """
    Class Callbacks attached with the application
    """

    def __init__(self):

        self.k_ref = 3
        self.pair = [None, None]
        self.rectangles = [None, None]
        self.line = None
        self.lines = []
        self.points = [[], []]

        self.rois = [None, None]
        self.fnames_tot = [None, None]

    def select_axis(self, event):
        """ Select the axis to be displayed in 'fig1' """
        control_key_pressed = event.guiEvent.modifiers() and Qt.ControlModifier

        if event.inaxes in self.model.ax or \
                (event.button in ['up', 'down'] and not control_key_pressed):

            if event.button == 'up':
                k = (self.k_ref + 1) % 4
            elif event.button == 'down':
                k = (self.k_ref - 1) % 4
            else:
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

    def init_or_remove_rectangle(self, event):
        """ Initialize or remove a rectangle """
        control_key_pressed = event.guiEvent.modifiers() and Qt.ControlModifier

        if self.toolbar.mode == '' and \
                event.inaxes == self.ax1 and \
                self.k_ref in [0, 1] and \
                control_key_pressed and \
                self.pair[0] is None:

            if event.button == 1:
                x, y = event.xdata, event.ydata
                self.pair = [[x, y], [None, None]]
                self.rectangles[self.k_ref] = Rectangle((x, y), 0, 0, ec='y', fc='none')
                self.ax1.add_patch(self.rectangles[self.k_ref])

            elif event.button == 3 and self.rectangles[self.k_ref] is not None:
                self.pair = [None, None]
                self.rectangles[self.k_ref].remove()
                self.rectangles[self.k_ref] = None
                self.rois_entry[self.k_ref].clear()
                self.update_rois(self.k_ref)
                self.canvas1.draw_idle()

    def draw_rectangle(self, event, set_roi=False):
        """ Draw the rectangle """
        control_key_pressed = event.guiEvent.modifiers() and Qt.ControlModifier

        if self.toolbar.mode == '' and \
                event.inaxes == self.ax1 and \
                self.k_ref in [0, 1] and \
                control_key_pressed and \
                self.pair[0] is not None:

            if event.button == 1:
                x, y = event.xdata, event.ydata
                x0, y0 = self.pair[0]
                self.rectangles[self.k_ref].set_width(x - x0)
                self.rectangles[self.k_ref].set_height(y - y0)
                self.canvas1.draw_idle()

            if set_roi:
                shape = self.model.imgs[self.k_ref].shape
                roi = [max(0, int(min(x0, x))), min(shape[1], int(max(x0, x))),
                       max(0, int(min(y0, y))), min(shape[0], int(max(y0, y)))]
                if roi[0] != roi[1] and roi[2] != roi[3]:
                    self.rois_entry[self.k_ref].clear()
                    self.rois_entry[self.k_ref].insert(str(roi))
                    self.update_rois(self.k_ref)
                self.pair = [None, None]

    def add_or_remove_points(self, event):
        """ Add or Remove a point """
        control_key_pressed = event.guiEvent.modifiers() and Qt.ControlModifier

        if self.toolbar.mode == '' and \
                event.inaxes == self.ax1 and \
                self.k_ref in [0, 1] and \
                self.group_reg_model.checkedButton().text() == 'User-Driven' and \
                not control_key_pressed:

            x, y = event.xdata, event.ydata
            model_points = self.model.points[self.k_ref]

            if event.button == 1:
                model_points.append([x, y])

            elif event.button == 3 and len(model_points) > 0:
                dist = [(xp - x) ** 2 + (yp - y) ** 2 for xp, yp in model_points]
                ind = dist.index(min(dist))
                del model_points[ind]

            [x.remove() for x in self.points[self.k_ref]]
            self.points[self.k_ref] = []

            for (x, y), color in zip(model_points, COLORS):
                self.points[self.k_ref].append(self.ax1.plot(x, y, 'o', mfc='none', color=color)[0])

            self.canvas1.draw_idle()

    def init_or_remove_line(self, event):
        """ Initialize or Remove a line """

        if self.toolbar.mode == '' and \
                event.inaxes == self.ax1 and \
                self.k_ref == 2 and \
                self.group_reg_model.checkedButton().text() == 'User-Driven':

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
                    x1p, y1p = x1 / rfacs[0], y1 / rfacs[0]
                    if alignment == 'horizontal':
                        x2p = (x2 - x12) / rfacs[1]
                        y2p = y2 / rfacs[1]
                    else:
                        x2p = x2 / rfacs[1]
                        y2p = (y2 - y12) / rfacs[1]
                    if self.model.rois[0] is not None:
                        x1p += self.model.rois[0][0]
                        y1p += self.model.rois[0][2]
                    if self.model.rois[1] is not None:
                        x2p += self.model.rois[1][0]
                        y2p += self.model.rois[1][2]
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
                self.group_reg_model.checkedButton().text() == 'User-Driven':

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

    def set_roi(self, event):
        """ Set ROI from the view to the model """
        self.draw_rectangle(event, set_roi=True)

    def zoom(self, event):
        """ Zoom/Unzoom the 'fig1' """
        control_key_pressed = event.guiEvent.modifiers() and Qt.ControlModifier

        if event.inaxes == self.ax1 and control_key_pressed:

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

    def update_params(self):
        """ Update the view parameters from the model and the plots """
        for k in range(2):
            self.rois_entry[k].clear()
            self.rois_entry[k].insert(str(self.model.rois[k]))
        ind = REG_MODELS.index(self.model.registration_model)
        self.group_reg_model.blockSignals(True)
        self.group_reg_model.button(ind).setChecked(True)
        self.group_reg_model.blockSignals(False)
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

        ind = fsel[k].list_widget.currentRow()
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
        self.model.binarized = self.check_binarized.isChecked()
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

        if self.check_binarized.isChecked():
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
            color = (255 * np.array(line.get_color())).astype(int)
            hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
            self.ax1.plot(line.get_xdata(), line.get_ydata(), c=hex_color, lw=2)

        if k_ref in [0, 1]:
            for rect in ax_ref.patches:
                self.rectangles[k_ref] = Rectangle(np.asarray(rect.get_xy()), rect.get_width(),
                                                   rect.get_height(), ec='y', fc='none')
                self.ax1.add_patch(self.rectangles[k_ref])
            for (x, y), color in zip(self.model.points[k_ref], COLORS):
                self.points[k_ref].append(self.ax1.plot(x, y, 'o', mfc='none', color=color)[0])

        if k_ref == 2:
            rfac = self.model.rfactors_plotting[0]
            shape = self.model.get_shapes()[0]
            if self.model.juxt_alignment == 'horizontal':
                self.ax1.axvline(shape[1] * rfac, c='w', ls='dashed', lw=0.5)
            elif self.model.juxt_alignment == 'vertical':
                self.ax1.axhline(shape[0] * rfac, c='w', ls='dashed', lw=0.5)

    def update_rois(self, k):
        """ Update ROIs of the k-th image from the Tkinter.Entry """
        roi_entry = self.rois_entry[k].text()
        if roi_entry == '':
            self.model.rois[k] = None
        else:
            try:
                self.model.rois[k] = eval(roi_entry)
            except:
                msg = f"{rois_entry}  cannot be interpreted as '[xmin, xmax, ymin, ymax]'"
                QMessageBox.critical(None, "", msg)
                return
        self.model.binarization_k(k)
        self.model.points = [[], []]
        self.model.img_reg = None
        self.model.img_reg_bin = None
        self.model.mask = None
        self.update_plots()

    def update_threshold(self, val, k):
        """ Update the threshold value associated with the k-th image """
        self.model.thresholds[k] = val / 100
        self.model.binarization_k(k)
        # self.model.registration()
        self.update_plots()

    def update_registration_model(self):
        """ Update the registration model value """
        self.model.registration_model = self.group_reg_model.checkedButton().text()
        self.model.reinit()
        self.update_plots()

    def update_max_size_reg(self):
        """ Update maximum image size for registration """
        self.model.max_size_reg = self.max_size_reg.value()

    def update_tmat_options(self, k):
        """ Update the tmat_options boolean values related to the 'key' """
        if self.tmat_options_cb[k].isChecked():
            for i, key_ in enumerate(REG_KEYS[:k + 1]):
                self.tmat_options_cb[i].setChecked(True)
                self.model.tmat_options[key_] = True
        else:
            for i, key_ in enumerate(REG_KEYS[k:]):
                self.tmat_options_cb[i + k].setChecked(False)
                self.model.tmat_options[key_] = False

    def update_interpolation(self):
        self.model.order = WARP_ORDERS[self.group_interpolations.checkedButton().text()]

    def update_inv_reg(self, val):
        """ Update the 'inv_reg' value """
        self.model.inv_reg = val
        self.model.registration_apply()
        self.update_plots(k=3)

    def update_fixed_reg(self, val):
        """ Update the 'fixed_reg' value """
        self.model.fixed_reg = val

    def update_juxt_alignment(self):
        """ Update the 'juxt_alignment' value """
        self.model.juxt_alignment = self.group_juxtaposition.checkedButton().text()
        self.update_plots(k=2)

    def update_apply_mask(self, val):
        """ Update the 'apply_mask' value """
        self.model.apply_mask = val
        self.update_plots(k=3)

    def update_angles(self, k):
        """ Update the angle of k-th image """
        self.model.angles[k] = int(self.group_angles[k].checkedButton().text())
        self.model.load_image(k, fname=self.model.fnames[k])
        self.update_plots()

    def update_resolution(self):
        """ Update the 'resolution' and 'rfactors_plotting' values """
        self.model.resolution = self.slider_resolution.value() / 100
        self.model.min_img_res = int(self.min_img_res.text())
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
        registration_model = self.group_reg_model.checkedButton().text()
        self.model.registration(registration_model=registration_model)
        self.update_plots()

    def save_images(self):
        """ Save the fixed and moving images in their final states """
        fnames_save = []
        for k in range(2):
            ind = self.fselectors[k].list_widget.currentRow()
            fname = Path(self.fselectors[k].fnames[ind])
            fname_save = str(fname.parent / (fname.stem + "_aligned" + fname.suffix))
            fnames_save.append(fd.getSaveFileName(self.window, "", fname_save)[0])
        self.model.save_images(fnames_save)

    def reload_params(self):
        """ Reload parameters """
        self.model.reload_params(obj=self.model)
        self.registration_model.set(self.model.registration_model)
        for k in range(2):
            self.thresholds[k].set(self.model.thresholds[k])
            self.rois_entry[k].clear()
            self.rois_entry[k].insert(str(self.model.rois[k]))
        self.update_plots()

    def apply_to_all(self, dirname_res=None):
        """ Apply the alignment processing to all the images """
        dirnames_res = [dirname_res]
        if self.model.dirname_res[0] is not None:
            dirnames_res.append(Path(self.model.dirname_res[0]).parent)
        if Path(self.model.fnames[0]).parents[1] in dirnames_res:
            msg = 'You are about to re-align images located in the RESULT dir.'
            if self.check_show_results.isChecked():
                msg += '\nTo go back to the raw data files, select "No" ' \
                       'and uncheck the "Show results" button.'
            msg += '\nContinue ?'
            reply = QMessageBox.question(None, "", msg,
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        self.model.apply_to_all(dirname_res=dirname_res)
        self.rois = self.model.rois
        self.fnames_tot = self.model.fnames_tot
        self.plot_results()

    def plot_results(self):
        """ Display the figures related to the 'dirname_res' """
        if self.model.dirname_res == [None, None]:
            return

        if self.check_show_results.isChecked():
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
