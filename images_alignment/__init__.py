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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from pystackreg import StackReg
from skimage.transform import warp, AffineTransform, estimate_transform

from images_alignment.utils import (Terminal,
                                    gray_conversion, image_normalization,
                                    resizing, cropping, padding, sift,
                                    concatenate_images, recast)

REG_MODELS = ['StackReg', 'SIFT', 'User-Driven']
STREG = StackReg(StackReg.AFFINE)
CMAP_BINARIZED = ListedColormap(["#00FF00", "black", "red"])
KEYS = ['areas', 'thresholds', 'bin_inversions', 'registration_model']

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
        self.areas = [None, None]
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

    def reinit(self):
        """ Reinitialize 'points', 'img_reg', 'img_reg_bin' and 'results' """
        self.points = [[], []]
        self.img_reg = None
        self.img_reg_bin = None
        self.results = {}

    def load_files(self, k, fnames):
        """ Load the k-th image files """
        if not isinstance(fnames, list):
            fnames = [fnames]

        self.fnames_tot[k] = fnames
        self.load_image(k, fnames[0])

    def load_image(self, k, fname):
        """ Load the k-th image """
        try:
            img = iio.imread(fname)
            self.reinit()
            self.fnames[k] = fname
            self.dtypes[k] = img.dtype
            self.imgs[k] = image_normalization(gray_conversion(img))
            self.binarization_k(k)

        except Exception as _:
            self.terminal.write(f"Failed to load {fname}\n\n")

    def set_area_k(self, k, area=None, area_percent=None):
        """ Set area parameter for the k-th image"""
        if area is not None and area_percent is not None:
            msg = "ERROR: 'area' and 'area_percent' cannot be defined " \
                  "simultaneously "
            self.terminal.write(msg)

        if area is not None:
            self.areas[k] = area
        elif area_percent is not None:
            xmin_p, xmax_p, ymin_p, ymax_p = area_percent
            shape = self.imgs[k].shape
            ymin, ymax = ymin_p * shape[0], ymax_p * shape[0]
            xmin, xmax = xmin_p * shape[1], xmax_p * shape[1]
            self.areas[k] = xmin, xmax, ymin, ymax

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

    def crop_and_resize(self, imgs):
        """ Crop and Resize the images"""
        imgs = [cropping(imgs[k], self.areas[k]) for k in range(2)]
        if self.registration_model == 'StackReg':
            imgs = resizing(*imgs)
        return imgs

    def registration(self, registration_model=None):
        """ Calculate the transformation matrix 'tmat' and apply it """
        self.registration_calc(registration_model=registration_model)
        self.registration_apply()

    def registration_calc(self, registration_model=None):
        """ Calculate the transformation matrix 'tmat' """
        if registration_model in REG_MODELS:
            self.registration_model = registration_model

        if self.registration_model == 'StackReg':
            imgs_bin = self.crop_and_resize(self.imgs_bin)
            self.tmat = STREG.register(*imgs_bin)

        elif self.registration_model == 'SIFT':
            imgs = self.crop_and_resize(self.imgs)
            self.tmat, self.points = sift(*imgs)

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

        imgs = self.crop_and_resize(self.imgs)
        imgs_bin = self.crop_and_resize(self.imgs_bin)

        output_shape = imgs[0].shape
        self.img_reg = warp(imgs[1], self.tmat,
                            output_shape=output_shape, preserve_range=True,
                            mode='constant', cval=1, order=None)
        self.img_reg_bin = warp(imgs_bin[1], self.tmat,
                                output_shape=output_shape, preserve_range=True,
                                mode='constant', cval=1, order=None)

        # score calculation
        mask = warp(np.ones_like(self.img_reg_bin), self.tmat, mode='constant',
                    cval=0, preserve_range=True, order=None)
        mismatch = np.logical_xor(imgs_bin[0], self.img_reg_bin)
        mismatch[~mask] = 0
        self.score = 100 * (1. - np.sum(mismatch) / np.sum(mask))

        self.results[self.registration_model] = {'score': self.score,
                                                 'tmat': self.tmat}

        return imgs[0], self.img_reg

    def set_dirname_res(self, dirname_res=None):
        """ Set dirname results 'dirname_res' """
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
            msg = f"ERROR: fixed images should be 1 or {n1} files.\n"
            msg += f"{n0} has been given\n\n"
            self.terminal.write(msg)
            return

        self.set_dirname_res(dirname_res=dirname_res)

        fnames_fixed = self.fnames_tot[0]
        fnames_moving = self.fnames_tot[1]
        for i, fname_moving in enumerate(fnames_moving):
            fname_fixed = fnames_fixed[0] if n0 == 1 else fnames_fixed[i]
            names = [Path(fname_fixed).name, Path(fname_moving).name]
            self.terminal.write(f"{i + 1}/{n1} {names[0]} - {names[1]}: ")

            try:

                self.load_image(0, fname=fname_fixed)
                self.load_image(1, fname=fname_moving)
                if not self.fixed_reg:
                    self.registration_calc()
                imgs = self.registration_apply()

                for k, img in enumerate(imgs):
                    iio.imwrite(self.dirname_res[k] / names[k],
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
    def reload_model(fname_json=None, obj=None):
        """ Reload model from a .json file and Return an ImagesAlign() object"""

        if fname_json is None:
            fname_json = filedialog.askopenfilename(defaultextension='.json')

        if not os.path.isfile(fname_json):
            raise IOError(f"{fname_json} is not a file")

        if obj is not None:
            assert isinstance(obj, ImagesAlign)

        with open(fname_json, 'r', encoding='utf-8') as fid:
            data = json.load(fid)

        imgalign = obj or ImagesAlign()
        for key, value in data.items():
            setattr(imgalign, key, value)
        return imgalign

    def plot_all(self, ax=None):
        """ Plot all the axis """
        if ax is not None:
            self.ax = ax

        for k in range(4):
            self.plot_k(k)

    def plot_k(self, k):
        """ Plot the k-th axis """
        self.ax[k].clear()

        if k in [0, 1]:
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

        self.ax[k].set_title(['Fixed image', 'Moving image'][k])

        if self.color == 'Binarized':
            img = np.zeros_like(self.imgs_bin[k], dtype=int)
            img[self.imgs_bin[k]] = 2 * k - 1
            self.ax[k].imshow(img, cmap=CMAP_BINARIZED, vmin=-1, vmax=1)
        else:
            self.ax[k].imshow(self.imgs[k], cmap='gray')

        if self.areas[k] is not None:
            xmin, xmax, ymin, ymax = self.areas[k]
            width, height = xmax - xmin, ymax - ymin
            self.ax[k].add_patch(Rectangle((xmin, ymin), width, height,
                                           ec='y', fc='none'))

    def plot_combined_images(self):
        """ Plot the combined images """

        self.ax[2].set_title("Combined images")

        if self.imgs[0] is None or self.imgs[1] is None:
            return

        if self.color == "Binarized":
            imgs = self.crop_and_resize(self.imgs_bin)
            if self.img_reg_bin is not None:
                imgs[1] = self.img_reg_bin.copy()
            imgs = padding(*imgs)
            img = np.zeros_like(imgs[0], dtype=int)
            img[imgs[1] * ~imgs[0]] = 1
            img[imgs[0] * ~imgs[1]] = -1
            self.ax[2].imshow(img, cmap=CMAP_BINARIZED, vmin=-1, vmax=1)

        else:
            imgs = self.crop_and_resize(self.imgs)
            if self.img_reg is not None:
                imgs[1] = self.img_reg.copy()
            imgs = padding(*imgs)
            img = 0.5 * (imgs[0] + imgs[1])
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

        x0 = y0 = x1 = y1 = 0
        if self.areas[0] is not None:
            x0, _, y0, _ = self.areas[0]
        if self.areas[1] is not None:
            x1, _, y1, _ = self.areas[1]

        rng = np.random.default_rng(0)
        for point0, point1 in zip(self.points[0][:30], self.points[1][:30]):
            color = rng.random(3)
            self.ax[3].plot((point0[0] + x0, point1[0] + x1 + offset[0]),
                            (point0[1] + y0, point1[1] + y1 + offset[1]), '-',
                            color=color)
