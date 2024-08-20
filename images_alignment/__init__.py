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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pystackreg import StackReg
from skimage.transform import resize, warp, AffineTransform, estimate_transform

from images_alignment.utils import (Terminal,
                                    gray_conversion, image_normalization,
                                    padding, sift,
                                    concatenate_images, recast)

REG_MODELS = ['StackReg', 'SIFT']
STREG = StackReg(StackReg.AFFINE)

STEP = 1  # default translation increment
ANGLE = np.deg2rad(0.5)  # default rotation angular increment
COEF = 1.01  # default scaling coefficient

AXES_NAMES = ['Fixed image', 'Moving image', 'Combined/Juxtaposed images']
COLORS = [(0, 1, 0), (0, 0, 0), (1, 0, 0)]  # Green -> Black -> Red
CMAP_BINARIZED = LinearSegmentedColormap.from_list('GreenBlackRed', COLORS, N=3)

KEYS = ['cropping_areas', 'thresholds', 'bin_inversions', 'tmat']

# plt.rcParams['image.origin'] = 'lower'
plt.rcParams['axes.titlesize'] = 10


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
    """

    def __init__(self, fnames_fixed=None, fnames_moving=None,
                 thresholds=None, bin_inversions=None, terminal=None):

        self.fnames_tot = [fnames_fixed, fnames_moving]
        self.fnames = [None, None]

        self.thresholds = thresholds or [0.5, 0.5]
        self.bin_inversions = bin_inversions or [False, False]
        self.terminal = terminal or Terminal()

        self.color = 'Gray'
        self.mode = 'Juxtaposed'

        self.imgs = [None, None]
        self.dtypes = [None, None]
        self.cropping_areas = [None, None]
        self.is_cropped = [False, False]
        self.imgs_bin = [None, None]
        self.registration_model = 'StackReg'
        self.points = [[], []]
        self.tmat = np.identity(3)
        self.score = 0
        self.img_reg = None
        self.img_reg_bin = None
        self.results = {}
        self.dirname_res = [None, None]
        self.fixed_reg = False

        _, ax = plt.subplots(2, 2, figsize=(8, 8))
        self.ax = ax.flatten()

        if self.fnames_tot[0] is not None:
            self.fnames[0] = self.fnames_tot[0]
            self.load_files(0, self.fnames[0])
        if self.fnames_tot[1] is not None:
            self.fnames[1] = self.fnames_tot[1]
            self.load_files(1, self.fnames[1])

    def reinit(self, k):
        """ Reinitialize the k-th image and beyond """
        self.imgs_bin[k] = None
        self.cropping_areas[k] = None
        self.is_cropped[k] = False
        self.tmat = None
        self.points = [[], []]
        self.img_reg = None
        self.results = {}

    def load_files(self, k, fnames):
        """ Load the k-th image files """
        if not isinstance(fnames, list):
            fnames = [fnames]

        self.fnames_tot[k] = fnames
        self.load_image(k, fnames[0], reinit=True)

    def load_image(self, k, fname, reinit=False):
        """ Load the k-th image """
        try:
            img = iio.imread(fname)
            if reinit:
                self.reinit(k)
            self.fnames[k] = fname
            self.imgs[k] = image_normalization(gray_conversion(img))
            self.dtypes[k] = img.dtype

        except Exception as _:
            self.terminal.write(f"Failed to load {fname}\n\n")

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
        elif area_percent is not None:
            xmin_p, xmax_p, ymin_p, ymax_p = area_percent
            shape = self.imgs[k].shape
            ymin, ymax = ymin_p * shape[0], ymax_p * shape[0]
            xmin, xmax = xmin_p * shape[1], xmax_p * shape[1]
            self.cropping_areas[k] = xmin, xmax, ymin, ymax

        if self.cropping_areas[k] is None:
            return
        else:
            xmin, xmax, ymin, ymax = self.cropping_areas[k]
            imin, imax, jmin, jmax = int(ymin), int(ymax), int(xmin), int(xmax)
            self.imgs[k] = self.imgs[k][imin:imax, jmin:jmax]
            self.is_cropped[k] = True

    def resizing(self):
        """ Resize the images to have similar shape (request for pyStackReg) """
        if self.imgs[0] is None or self.imgs[1] is None:
            return

        if self.imgs[0].size <= self.imgs[1].size:
            self.imgs[0] = resize(self.imgs[0], self.imgs[1].shape)
        else:
            self.imgs[1] = resize(self.imgs[1], self.imgs[0].shape)

    def binarization_k(self, k):
        """ Binarize the k-th image """
        if self.imgs[k] is None:
            return

        self.imgs_bin[k] = self.imgs[k] > self.thresholds[k]

        if self.bin_inversions[k]:
            self.imgs_bin[k] = ~self.imgs_bin[k]

    def binarization(self):
        """ Binarize the images """
        self.binarization_k(0)
        self.binarization_k(1)

    def registration(self, registration_model=None):
        """ Calculate the transformation matrix 'tmat' and apply it """
        self.registration_calc(registration_model=registration_model)
        self.registration_apply()

    def registration_calc(self, registration_model=None):
        """ Calculate the transformation matrix 'tmat' """
        if registration_model in REG_MODELS:
            self.registration_model = registration_model

        if self.registration_model == 'StackReg':
            self.resizing()
            self.binarization_k(0)  # reinit
            self.binarization_k(1)  # reinit
            self.tmat = STREG.register(*self.imgs_bin)

        elif self.registration_model == 'SIFT':
            self.tmat, self.points = sift(*self.imgs)

        elif self.registration_model == 'User-Driven':
            src = np.asarray(self.points[0])
            dst = np.asarray(self.points[1])
            self.tmat = estimate_transform('affine', src, dst).params

        else:
            raise IOError

        print()
        print(self.tmat)

    def registration_apply(self):
        """ Apply the transformation matrix 'tmat' to the moving image """
        if self.tmat is None:
            return

        self.binarization_k(0)  # reinit
        self.binarization_k(1)  # reinit

        output_shape = self.imgs[0].shape
        self.img_reg = warp(self.imgs[1], self.tmat,
                            output_shape=output_shape, preserve_range=True,
                            mode='constant', cval=1, order=None)
        self.img_reg_bin = warp(self.imgs_bin[1], self.tmat,
                                output_shape=output_shape, preserve_range=True,
                                mode='constant', cval=1, order=None)

        # score calculation
        mask = warp(np.ones_like(self.img_reg_bin), self.tmat, mode='constant',
                    cval=0, preserve_range=True, order=None)
        mismatch = np.logical_xor(self.imgs_bin[0], self.img_reg_bin)
        mismatch[~mask] = 0
        self.score = 100 * (1. - np.sum(mismatch) / np.sum(mask))

        self.results[self.registration_model] = {'score': self.score,
                                                 'tmat': self.tmat}

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

    def rotate(self, angle=ANGLE, xc_rel=0.5, yc_rel=0.5):
        """ Apply rotation coefficients in 'tmat' """

        rotation_mat = np.array([[np.cos(angle), np.sin(angle), 0],
                                 [-np.sin(angle), np.cos(angle), 0],
                                 [0, 0, 1]])

        shape = self.imgs_bin[0].shape
        transl_x, transl_y = int(xc_rel * shape[1]), int(yc_rel * shape[0])

        transl = np.array([[1, 0, -transl_x],
                           [0, 1, -transl_y],
                           [0, 0, 1]])
        inv_transl = np.array([[1, 0, transl_x],
                               [0, 1, transl_y],
                               [0, 0, 1]])

        self.tmat = self.tmat @ inv_transl @ rotation_mat @ transl
        self.registration_apply()

    def set_dirname_res(self, dirname_res=None):
        if dirname_res is None:
            initialdir = None
            if self.fnames_tot[1] is not None:
                initialdir = Path(self.fnames_tot[1][-1]).parent
            dirname_res = filedialog.askdirectory(initialdir=initialdir)
            if dirname_res is None:
                return

        dirname_res = Path(dirname_res)
        dirname_res.mkdir(exist_ok=True)

        self.dirname_res[0] = dirname_res / "fixed_images"
        self.dirname_res[1] = dirname_res / "moving_images"
        self.dirname_res[0].mkdir(exist_ok=True)
        self.dirname_res[1].mkdir(exist_ok=True)

    def apply_to_all(self, dirname_res=None):
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

        self.set_dirname_res(dirname_res=dirname_res)

        fnames_fixed = self.fnames_tot[0]
        fnames_moving = self.fnames_tot[1]
        cropping_areas = self.cropping_areas.copy()

        for i, fname_moving in enumerate(fnames_moving):

            fname_fixed = fnames_fixed[0] if n0 == 1 else fnames_fixed[i]

            names = [Path(fname_fixed).name, Path(fname_moving).name]

            self.terminal.write(f"{i + 1}/{n1} {names[0]} - {names[1]}: ")

            try:

                self.load_image(0, fname=fname_fixed, reinit=True)
                self.load_image(1, fname=fname_moving, reinit=True)
                self.cropping(0, area=cropping_areas[0])
                self.cropping(1, area=cropping_areas[1])
                if not self.fixed_reg:
                    self.registration_calc()
                self.registration_apply()

                for k, img in enumerate([self.imgs[0], self.img_reg]):
                    iio.imwrite(self.dirname_res[k] / name[k],
                                recast(img, self.dtypes[k]))

                score = self.results[self.registration_model]['score']
                self.terminal.write(f"OK - score : {score:.1f} %\n")

            except:
                self.terminal.write("FAILED\n")

        self.terminal.write("\n")

    def save_model(self, fname_json=None):
        """ Save model in a .json file """
        if fname_json is None:
            fname_json = filedialog.asksaveasfilename(defaultextension='.json')
            if fname_json is None:
                return

        data = {}
        for key in KEYS:
            data.update({key: eval(f"self.{key}")})
        data.update({'tmat': self.tmat.tolist()})

        with open(fname_json, 'w', encoding='utf-8') as fid:
            json.dump(data, fid, ensure_ascii=False, indent=4)

    @staticmethod
    def reload_model(fname_json=None):
        """ Reload model from a .json file and Return an ImagesAlign() object"""

        if fname_json is None:
            fname_json = filedialog.askopenfile(defaultextension='.json')

        if isinstance(fname_json, (str, Path)) and os.path.isfile(fname_json):
            with open(fname_json, 'r', encoding='utf-8') as fid:
                data = json.load(fid)

            imgalign = ImagesAlign()
            for key, value in data.items():
                setattr(imgalign, key, value)
            imgalign.tmat = np.asarray(imgalign.tmat)
            return imgalign
        else:
            return None

    def plot_all(self, ax=None):
        """ Plot all the axis """
        if ax is not None:
            self.ax = ax

        for k in range(4):
            self.plot_k(k)

    def plot_k(self, k):
        """ Plot the k-th axis """
        self.ax[k].clear()

        if k == 0 or k == 1:
            self.plot_fixed_or_moving_image(k)

        elif k == 2:
            self.plot_combined_images()

        elif k == 3:
            self.plot_juxtaposed_images()

        else:
            raise IOError

        self.ax[k].autoscale(tight=True)

    def plot_fixed_or_moving_image(self, k):
        """ Plot the fixed or the moving image """

        if self.imgs[k] is None:
            return

        title = AXES_NAMES[k]
        # title += f" - {Path(self.fnames[k]).name}"

        self.ax[k].set_title(title)

        if self.color == 'Binarized':
            if self.imgs_bin[k] is None:
                self.binarization_k(k)
            img = np.zeros_like(self.imgs_bin[k], dtype=int)
            img[self.imgs_bin[k]] = 2 * k - 1
            self.ax[k].imshow(img, cmap=CMAP_BINARIZED, vmin=-1, vmax=1)
        else:
            # if k==1 and self.img_reg is not None:
            self.ax[k].imshow(self.imgs[k], cmap='gray')

    def plot_combined_images(self):
        """ Plot the combined images """

        self.ax[2].set_title("Combined images")

        if self.imgs[0] is None or self.imgs[1] is None:
            return

        if self.color == "Binarized":
            if self.imgs_bin[0] is None:
                self.binarization_k(0)
            if self.imgs_bin[1] is None:
                self.binarization_k(1)
            img_0, img_1 = self.imgs_bin
            if self.img_reg_bin is not None:
                img_1 = self.img_reg_bin
            img_0, img_1 = padding(img_0, img_1)
            img = np.zeros_like(img_0, dtype=int)
            img[img_1 * ~img_0] = 1
            img[img_0 * ~img_1] = -1
            self.ax[2].imshow(img, cmap=CMAP_BINARIZED, vmin=-1, vmax=1)

        else:
            img_0, img_1 = self.imgs
            if self.img_reg is not None:
                img_1 = self.img_reg
            img_0, img_1 = padding(img_0, img_1)
            img = 0.5 * (img_0 + img_1)
            self.ax[2].imshow(img, cmap='gray')

    def plot_juxtaposed_images(self):
        """ Plot the juxtaposed images """

        self.ax[3].set_title("Juxtaposed images")

        img_0 = self.ax[0].get_images()
        img_1 = self.ax[1].get_images()
        if len(img_0) == 0 or len(img_1) == 0:
            return

        arr_0 = img_0[0].get_array()
        arr_1 = img_1[0].get_array()

        img, offset = concatenate_images(arr_0, arr_1, 'horizontal')

        if self.color == 'Gray':
            self.ax[3].imshow(img, cmap='gray')
        else:
            self.ax[3].imshow(img, cmap=CMAP_BINARIZED, vmin=-1, vmax=1)

        rng = np.random.default_rng(0)
        for point0, point1 in zip(self.points[0][:30], self.points[1][:30]):
            color = rng.random(3)
            self.ax[3].plot((point0[0], point1[0] + offset[0]),
                            (point0[1], point1[1] + offset[1]), '-',
                            color=color)
