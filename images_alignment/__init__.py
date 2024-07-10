"""
Application for images registration
"""
import os
from copy import deepcopy
from pathlib import Path
import json
from tkinter import filedialog
import panel as pn
import numpy as np
import imageio.v3 as iio
from matplotlib.figure import Figure
from pystackreg import StackReg
from skimage.transform import warp, AffineTransform
from skimage.feature import plot_matches

from images_alignment.utils import (gray_conversion, image_normalization,
                                    edges_trend, padding, interpolation, sift)

REG_MODELS = ['StackReg', 'SIFT']
STREG = StackReg(StackReg.AFFINE)

STEP = 1  # translation increment
ANGLE = np.deg2rad(1)  # rotation angular increment
COEF1, COEF2 = 0.99, 1.01  # scaling coefficient

KEYS = ['fnames_fixed', 'fnames_moving',
        'thresholds', 'bin_inversions', 'mode_auto', 'tmat']


class ImagesAlign:
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

        self.fnames_tot = [fnames_fixed, fnames_moving]
        self.fnames = [None, None]

        self.thresholds = thresholds or [0.5, 0.5]
        self.bin_inversions = bin_inversions or [False, False]
        self.mode_auto = mode_auto

        self.imgs = [None, None]
        self.cropping_areas = [None, None]
        self.is_cropped = [False, False]
        self.imgs_bin = [None, None]
        self.registration_model = 'StackReg'
        self.keypoints = [[], []]
        self.descriptors = [[], []]
        self.matches = None
        self.tmat = np.identity(3)
        self.img_reg = None
        self.results = {}
        self.dir_results = None
        self.fixed_reg = False

        self.input_fnames = None
        self.save_fname = None
        self.reload_fname = None

        if self.fnames_tot[0] is not None:
            self.fnames[0] = self.fnames_tot[0][0]
            self.load_files(0, self.fnames[0])
        if self.fnames_tot[1] is not None:
            self.fnames[1] = self.fnames_tot[1][0]
            self.load_files(1, self.fnames[1])

    def reinit(self, k):
        """ Reinitialize the k-th image and beyond """
        self.imgs[k] = self.imgs_bin[k] = None
        self.cropping_areas[k] = None
        self.is_cropped[k] = False
        self.tmat = self.keypoints = self.descriptors = self.matches = None
        self.img_reg = None
        self.results = {}

    def load_files(self, k, fnames):
        """ Load the k-th image files """
        if not isinstance(fnames, list):
            fnames = [fnames]
        try:
            img = iio.imread(fnames[0])
        except Exception as _:
            self.terminal.write(f"Failed to load {fnames[0]}\n\n")
            return

        self.fnames_tot[k] = fnames
        self.fnames[k] = fnames[0]

        # image normalization in range [0, 1]
        self.imgs[k] = image_normalization(gray_conversion(img))

    def cropping(self, k, area=None, area_percent=None):
        """ Crop the k-th image"""
        if area is not None and area_percent is not None:
            msg = "ERROR: 'area' and 'area_percent' cannot be defined " \
                  "simultaneously "
            self.terminal.write(msg)

        if self.is_cropped[k]:
            msg = "ERROR: 2 consecutive crops are not allowed. "
            msg += "Please, REINIT the image\n"
            self.terminal.write(msg)
            return

        if area is not None:
            self.cropping_areas[k] = area
        if area_percent is not None:
            xmin, xmax, ymin, ymax = area_percent
            shape = self.imgs[k].shape
            imin, imax = int(ymin * shape[0]), int(ymax * shape[0])
            jmin, jmax = int(xmin * shape[1]), int(xmax * shape[1])
            self.cropping_areas[k] = [imin, imax, jmin, jmax]

        imin, imax, jmin, jmax = self.cropping_areas[k]
        self.imgs[k] = self.imgs[k][imin:imax, jmin:jmax]
        self.is_cropped[k] = True

        if self.mode_auto:
            self.resizing()

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

    def binarization_k(self, k):
        """ Binarize the k-th image """
        img_bin = self.imgs[k] > self.thresholds[k]
        if edges_trend(self.imgs[k]):
            img_bin = ~img_bin
        if self.bin_inversions[k]:
            img_bin = ~img_bin
        return img_bin

    def binarization(self):
        """ Binarize the images """
        self.imgs_bin = [self.binarization_k(0), self.binarization_k(1)]

        if self.mode_auto:
            self.registration()

    def registration(self, registration_model=None):
        self.registration_calc(registration_model=registration_model)
        self.registration_apply()

    def registration_calc(self, registration_model=None):
        """ Calculate the transformation matrix 'tmat' and apply it """
        if registration_model in REG_MODELS:
            self.reg_models.value = registration_model

        if self.registration_model == 'StackReg':
            self.imgs_bin[1] = self.binarization_k(1)  # reinit
            self.tmat = STREG.register(*self.imgs_bin)

        elif self.registration_model == 'SIFT':
            out = sift(*self.imgs)
            self.tmat, self.keypoints, self.descriptors, self.matches = out

        else:
            raise IOError

        print(self.tmat)

    def registration_apply(self):
        """ Apply 'tmat' to the moving image """

        self.imgs_bin[0] = self.binarization_k(0)  # reinit
        self.imgs_bin[1] = self.binarization_k(1)  # reinit

        self.img_reg = warp(self.imgs[1], self.tmat, mode='constant',
                            cval=1, preserve_range=True, order=None)
        self.imgs_bin[1] = warp(self.imgs_bin[1], self.tmat, mode='constant',
                                cval=1, preserve_range=True, order=None)

        # score calculation
        mask = warp(np.ones_like(self.imgs_bin[1]), self.tmat, mode='constant',
                    cval=0, preserve_range=True, order=None)
        mismatch = np.logical_xor(*self.imgs_bin)
        mismatch[~mask] = 0
        score = 100 * (1. - np.sum(mismatch) / np.sum(mask))

        self.results[self.registration_model] = {'score': score,
                                                 'tmat': self.tmat}

        np.set_printoptions(precision=4)
        self.result_str.object = f'SCORE: {score:.1f} % \n\n {self.tmat}'
        np.set_printoptions(precision=None)

        [self.update_plot(i) for i in range(3)]
        self.update_plot_zoom()

    def translate(self, mode, step=STEP):
        """ Apply translation step in 'tmat' """
        if mode == 'up':
            self.tmat[1, 2] -= step
        elif mode == 'down':
            self.tmat[1, 2] += step
        elif mode == 'left':
            self.tmat[0, 2] += step
        elif mode == 'right':
            self.tmat[0, 2] -= step
        self.registration_apply()

    def rotate(self, angle=ANGLE, reverse=False):
        """ Apply rotation coefficients in 'tmat' """

        if reverse:
            angle *= -1

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
        self.registration_apply()

    def select_dir_result(self):
        dirname = filedialog.askdirectory()
        if dirname:
            self.dir_results = dirname

    def apply(self, dirname_res=None):
        """ Apply the transformations to a set of images """
        if self.fnames_tot[0] is None:
            self.terminal.write("ERROR: fixed images are not defined\n\n")
            return
        if self.fnames_tot[1] is None:
            self.terminal.write("ERROR: moving images are not defined\n\n")
            return

        n0, n1 = len(self.fnames_tot[0]), len(self.fnames_tot[1])
        if not (n0 == 1 or n0 != n1):
            msg = f"ERROR: fixed images should consist in 1 or {n1} files.\n"
            msg += f"{n0} has been given\n\n"
            self.terminal.write(msg)
            return

        if dirname_res is None:
            initialdir = Path(self.fnames_tot[1][-1]).parent
            dirname_res = filedialog.askdirectory(initialdir=initialdir)
            if dirname_res is None:
                return
        dirname_res = Path(dirname_res)
        dirname_res0 = dirname_res / "fixed_images"
        dirname_res1 = dirname_res / "moving_images"
        dirname_res.mkdir(exist_ok=True)
        dirname_res0.mkdir(exist_ok=True)
        dirname_res1.mkdir(exist_ok=True)

        fnames_fixed = self.fnames_tot[0]
        fnames_moving = self.fnames_tot[1]

        mode_auto_save = self.mode_auto
        self.mode_auto = False

        for i, fname_moving in enumerate(fnames_moving):

            fname_fixed = fnames_fixed[0] if n0 == 1 else fnames_fixed[i]

            self.fnames = [fname_fixed, fname_moving]

            name0 = Path(fname_fixed).name
            name1 = Path(fname_moving).name

            self.terminal.write(f"{i + 1}/{n1} {name0} - {name1}: ")

            try:

                self.imgs[0] = iio.imread(fname_fixed)
                self.imgs[1] = iio.imread(fname_moving)

                if self.cropping_areas[0] is not None:
                    (imin, imax, jmin, jmax) = self.cropping_areas[0]
                    self.imgs[0] = self.imgs[0][imin: imax, jmin: jmax]

                if self.cropping_areas[1] is not None:
                    (imin, imax, jmin, jmax) = self.cropping_areas[1]
                    self.imgs[1] = self.imgs[1][imin: imax, jmin: jmax]

                self.resizing()
                if i == 0 or not self.fixed_reg:
                    self.registration_calc()
                self.registration_apply()

                if n0 != 1 or i == 0:
                    iio.imwrite(dirname_res0 / name0, self.imgs[0])
                iio.imwrite(dirname_res1 / name1, self.img_reg)

                score = self.results[self.registration_model]['score']
                self.terminal.write(f"OK - score : {score:.1f} %\n")

            except:
                self.terminal.write("FAILED\n")

        self.terminal.write("\n")
        self.mode_auto = mode_auto_save

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

            app = ImagesAlign(fnames_fixed=data['fnames_fixed'],
                              fnames_moving=data['fnames_moving'],
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
