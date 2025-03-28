"""
Class dedicated to images alignment
"""
import os
import shutil
import warnings
from pathlib import Path
import json
from tkinter import filedialog
from tempfile import gettempdir
import random
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from pystackreg import StackReg
from skimage.transform import warp, estimate_transform

from images_alignment.utils import (Terminal, fnames_multiframes_from_list,
                                    gray_conversion, imgs_conversion,
                                    rescaling_factors, imgs_rescaling,
                                    image_normalization, absolute_threshold,
                                    resizing, cropping, padding, sift,
                                    concatenate_images, rescaling, get_transformation)

TMP_DIR = Path(gettempdir()) / "images_alignment"
shutil.rmtree(TMP_DIR, ignore_errors=True)
os.makedirs(TMP_DIR, exist_ok=True)

REG_MODELS = ['StackReg', 'SIFT', 'SIFT + StackReg', 'User-Driven']
REG_KEYS = ['translation', 'rotation', 'scaling', 'shearing']
WARP_ORDERS = {'Default': None, 'Nearly': 0, 'Linear': 1}
CMAP_BINARIZED = ListedColormap(["#00FF00", "black", "red"])
KEYS = ['rois', 'thresholds', 'bin_inversions', 'registration_model']
COLORS = plt.cm.tab10.colors

plt.rcParams['axes.titlesize'] = 10


class ImagesAlign:
    """
    Class dedicated to images alignment

    Parameters
    ----------
    fnames_fixed, fnames_moving: iterables of str, optional
        Images pathnames related to fixed and moving images resp. to handle
    rois: list of 2 iterables, optional
        rois (regions of interest) attached to the fixed and moving images, each defining as:
         [xmin, xmax, ymin, ymax]
    thresholds: iterable of 2 floats, optional
        Thresholds used to binarize the images. Default values are [0.5, 0.5]
    bin_inversions: iterable of 2 bools, optional
        Activation keywords to reverse the image binarization
    terminal: object, optional
        object with an associated write() class function to write messages

    Attributes
    ----------
    fnames_tot: list of 2 list
        Images pathnames related to fixed and moving images resp.
    fnames: list of 2 str
        List of the current fixed and moving images filenames
    imgs, imgs_bin: list of 2 arrays
        Arrays related to the current fixed and moving images, in their raw or binarized states
        resp.
    dtype: list of 2 dtype
        Datatypes associated to the current fixed and moving images
    rois: list of 2 iterables, optional
        rois (regions of interest) attached to the fixed and moving images, each defining as:
         [xmin, xmax, ymin, ymax]
    thresholds: iterable of 2 floats
        Thresholds used to binarize the images. Default values are [0.5, 0.5]
    angles: list of 2 int
        angles to apply resp to the fixed and moving images before registration.
        (each angle to be chosen among [0, 90, 180, 270])
    bin_inversions: iterable of 2 bools
        Activation keywords to reverse the image binarization
    terminal: object
        object with an associated write() class function to write messages
    registration_model: str
        Registration model to be considered among ['StackReg', 'SIFT', 'SIFT + StackReg',
        'User-Driven'].
        Default is 'StackReg'.
    points: list of 2 list of floats
        Coordinates of the matching points between the fixed and moving images (filled by all
        registration models except 'StackReg').
    max_size_reg: int
        Maximum image size considered when performing the registration.
        Default value is 512.
    inv_reg: bool
        Activation key to reverse 'fixed' and 'moving' images during the registration calculation.
        Default is False.
    tmat: numpy.ndarray((3, 3))
        Transformation matrice returned by the registration.
    img_reg, img_reg_bin: arrays
        Arrays issued from the moving registrated image in its raw or binarized state resp.
    mask: numpy.ndarray
        Array (dtype=bool) associated with the overlapping area issued from the registration.
    tmat_options: dictionary of bool, optional
        Dictionary composed of options to build the 'tmat' transformation matrice (translation,
        rotation, scaling, shearing). Default values are True.
    order: int, optional
        The order of interpolation. See:
        https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp
    fixed_reg: bool
        Activation key to perform the worflow calculation with a fixed 'tmat'.
        Default is False.
    results: dict
        Dictionary that contains the workflow score and tmat of each alignment
    dirname_res: list of 2 str
        Filenames associated with the 'fixed_images' and 'moving_images' resulting images
        sub-folders
    binarized: bool
        Activation key to display images according to their binarized values
    resolution: float
        relative resolution parameter (between [0., 1.]).
        Default value is 0.
    min_img_res: int
        Minimum image resolution associated to 'resolution'=0.
        Default value is 256.
    rfactors_plotting: list of 2 floats
        Rescaling factors used for images displaying (each one between [0., 1.]).
        Default values are [1., 1.].
    juxt_alignment: str
        Mode of images juxtapostion ('horizontal' or 'vertical').
        Default value is 'horizontal'.
    apply_mask: bool
        Activation keyword to limit the combined image to the overlapping area.
        Default is True.
    ax: matplotlib.axes.Axes object
        Axes associated to the image alignment application to display the different images.
        Default is a 4 horizontal axes to respectively display the 'fixed' and 'moving' image,
        the 'juxtaposed image' and the 'combined image'.
    """

    def __init__(self, fnames_fixed=None, fnames_moving=None, rois=None,
                 thresholds=None, bin_inversions=None, terminal=None):

        if fnames_fixed is not None:
            fnames_fixed = fnames_multiframes_from_list(fnames_fixed)
        if fnames_moving is not None:
            fnames_moving = fnames_multiframes_from_list(fnames_moving)

        self.fnames_tot = [fnames_fixed, fnames_moving]
        self.fnames = [None, None]
        self.imgs = [None, None]
        self.imgs_bin = [None, None]
        self.dtypes = [None, None]

        # calculation parameters
        self.rois = rois or [None, None]
        self.angles = [0, 0]
        self.thresholds = thresholds or [0.5, 0.5]
        self.bin_inversions = bin_inversions or [False, False]
        self.terminal = terminal or Terminal()

        # 'registration' parameters and attributes
        self.registration_model = 'StackReg'
        self.points = [[], []]
        self.max_size_reg = 512
        self.inv_reg = False
        self.tmat = np.identity(3)
        self.img_reg = None
        self.img_reg_bin = None
        self.mask = None
        self.tmat_options = {key: True for key in REG_KEYS}
        self.order = None

        # 'application' attributes
        self.fixed_reg = False
        self.results = {}
        self.dirname_res = [None, None]

        # visualization parameters
        self.binarized = False
        # self.mode = 'Juxtaposed'
        self.resolution = 0.
        self.min_img_res = 256
        self.rfactors_plotting = [1., 1.]
        self.juxt_alignment = 'horizontal'
        self.apply_mask = True

        _, self.ax = plt.subplots(1, 4, figsize=(10, 4),
                                  gridspec_kw={'width_ratios': [1, 1, 1, 2]})

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
        self.mask = None

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
            success = True
        except Exception as _:
            self.terminal.write(f"Failed to load {fname}\n\n")
            success = False

        if success:
            self.reinit()
            self.imgs[k] = np.rot90(img, k=self.angles[k] / 90)
            self.dtypes[k] = img.dtype
            self.fnames[k] = fname
            self.binarization_k(k)
            self.update_rfactors_plotting()

    def get_shapes(self):
        """ Return the shapes related to the cropped (or not cropped) images """
        shapes = []
        for k in range(2):
            if self.rois[k] is not None:
                xmin, xmax, ymin, ymax = self.rois[k]
                shapes.append((ymax - ymin, xmax - xmin))
            else:
                shape = self.imgs[k].shape
                shapes.append([shape[0], shape[1]])
        return shapes

    def update_rfactors_plotting(self):
        """ Update the 'rfactors_plotting' wrt 'resolution' and 'imgs' sizes """
        if self.imgs[0] is None or self.imgs[1] is None:
            return

        imgs = [cropping(self.imgs[k], self.rois[k], verbosity=False) for k in range(2)]
        shapes = [imgs[0].shape[:2], imgs[1].shape[:2]]
        vmax = max(max(shapes[0]), max(shapes[1]))
        vmin = min(self.min_img_res, max(shapes[0]), max(shapes[1]))
        max_size = (vmax - vmin) * self.resolution + vmin
        self.rfactors_plotting = rescaling_factors(imgs, max_size)

    def binarization_k(self, k):
        """ Binarize the k-th image """
        if self.imgs[k] is None:
            return

        img = image_normalization(gray_conversion(self.imgs[k]))
        abs_threshold = absolute_threshold(cropping(img, self.rois[k]), self.thresholds[k])
        self.imgs_bin[k] = img > abs_threshold

        if self.bin_inversions[k]:
            self.imgs_bin[k] = ~self.imgs_bin[k]

    def binarization(self):
        """ Binarize the images """
        self.binarization_k(0)
        self.binarization_k(1)

    def crop_and_resize(self, imgs, verbosity=True):
        """ Crop and Resize the images"""
        imgs = [cropping(imgs[k], self.rois[k], verbosity=verbosity) for k in range(2)]
        if self.registration_model == 'StackReg':
            imgs = resizing(*imgs)
        return imgs

    def registration(self, registration_model=None, show_score=True):
        """ Calculate the transformation matrix 'tmat' and apply it """
        self.registration_calc(registration_model=registration_model)
        self.registration_apply(show_score=show_score)

    def registration_calc(self, registration_model=None):
        """ Calculate the transformation matrix 'tmat' """
        if registration_model in REG_MODELS:
            self.registration_model = registration_model

        transformation = get_transformation(self.tmat_options, self.registration_model)

        if self.registration_model == 'StackReg':
            imgs_bin = self.crop_and_resize(self.imgs_bin)
            imgs_bin, rfacs = imgs_rescaling(imgs_bin, self.max_size_reg)
            self.tmat = StackReg(eval(f"StackReg.{transformation}")).register(*imgs_bin)
            self.tmat[:2, :2] *= rfacs[0] / rfacs[1]
            self.tmat[:2, 2] *= 1. / rfacs[1]

        elif self.registration_model == 'SIFT':
            imgs = self.crop_and_resize(self.imgs)
            imgs = [gray_conversion(img) for img in imgs]
            imgs, rfacs = imgs_rescaling(imgs, self.max_size_reg)
            self.tmat, self.points = sift(*imgs, model_class=transformation)
            self.tmat[:2, :2] *= rfacs[0] / rfacs[1]
            self.tmat[:2, 2] *= 1. / rfacs[1]
            if len(self.points[0]) > 0:
                self.points[0] = self.points[0] / rfacs[0]
                self.points[1] = self.points[1] / rfacs[1]
            if self.rois[0] is not None:
                self.points[0][:, :] += self.rois[0][::2]
            if self.rois[1] is not None:
                self.points[1][:, :] += self.rois[1][::2]

        elif self.registration_model == 'User-Driven':
            if 0 < len(self.points[1]) == len(self.points[0]):
                src = np.asarray(self.points[0])
                dst = np.asarray(self.points[1])

                if self.rois[0] is not None:
                    src[:, :] -= self.rois[0][::2]  # remove (xmin, ymin)
                    src[:, 1] = self.rois[0][3] - self.rois[0][2] - src[:, 1]  # inverse y-axis
                else:
                    src[:, 1] = self.imgs[0].shape[0] - src[:, 1]  # inverse y-axis

                if self.rois[1] is not None:
                    dst[:, :] -= self.rois[1][::2]  # remove (xmin, ymin)
                    dst[:, 1] = self.rois[1][3] - self.rois[1][2] - dst[:, 1]  # inverse y-axis
                else:
                    dst[:, 1] = self.imgs[1].shape[0] - dst[:, 1]  # inverse y-axis

                transformation.estimate(src, dst)
                self.tmat = transformation.params
            else:
                self.tmat = np.eye(3)

        elif self.registration_model == 'SIFT + StackReg':

            self.registration(registration_model='SIFT', show_score=False)

            # save/change input data
            tmat = self.tmat.copy()
            imgs = self.imgs.copy()
            imgs_bin = self.imgs_bin.copy()
            rois = self.rois.copy()

            # change temporarily input data
            self.imgs[1] = self.img_reg
            self.imgs[0] = cropping(self.imgs[0], self.rois[0]).astype(float)
            self.imgs[0][self.mask] = np.nan
            self.rois = [None, None]
            self.binarization()

            self.registration_calc(registration_model='StackReg')
            self.tmat = np.matmul(tmat, self.tmat)

            # re-set data to their original values
            self.registration_model = 'SIFT + StackReg'
            self.imgs = imgs
            self.imgs_bin = imgs_bin
            self.rois = rois

        else:
            raise IOError

        print()
        print(self.tmat)

    def registration_apply(self, show_score=True):
        """ Apply the transformation matrix 'tmat' to the moving image """
        if self.tmat is None:
            return

        imgs = self.crop_and_resize(self.imgs, verbosity=False)
        imgs_bin = self.crop_and_resize(self.imgs_bin, verbosity=False)

        k0, k1, tmat = 0, 1, self.tmat
        if self.inv_reg:  # inverse registration from the fixed to the moving image
            k0, k1, tmat = 1, 0, np.linalg.inv(self.tmat)

        output_shape = imgs[k0].shape
        self.img_reg = warp(imgs[k1], tmat,
                            output_shape=output_shape,
                            preserve_range=True,
                            mode='constant', cval=np.nan, order=self.order)
        self.img_reg_bin = warp(imgs_bin[k1], tmat,
                                output_shape=output_shape[:2])

        self.mask = np.isnan(self.img_reg)
        if len(self.mask.shape) > 2:
            self.mask = self.mask.any(axis=-1)

        # score calculation and displaying
        if show_score:
            mismatch = np.logical_xor(imgs_bin[k0], self.img_reg_bin)
            mismatch[self.mask] = 0
            delta = mismatch.size - np.sum(self.mask)
            score = 100 * (1. - np.sum(mismatch) / delta) if delta != 0 else 0
            msg = f"score : {score:.1f} % ({self.registration_model}"
            if "SIFT" in self.registration_model:
                msg += f" - nb_matches : {len(self.points[0])}"
            msg += ")"
            self.terminal.write(msg + "\n")

            self.results[self.registration_model] = {'score': score,
                                                     'tmat': self.tmat}

        return imgs[0], self.img_reg

    def save_images(self, fnames_save):
        """ Save the fixed and moving images in their final states """
        imgs = self.crop_and_resize(self.imgs, verbosity=False)
        if self.img_reg is not None:
            k0 = 0 if self.inv_reg else 1
            imgs[k0] = self.img_reg

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            [iio.imwrite(fname, imgs[k].astype(self.dtypes[k]))
             for k, fname in enumerate(fnames_save) if fname != '']

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
            self.terminal.write("\nERROR: fixed images are not defined\n\n")
            return
        if self.fnames_tot[1] is None:
            self.terminal.write("\nERROR: moving images are not defined\n\n")
            return

        n0, n1 = len(self.fnames_tot[0]), len(self.fnames_tot[1])
        if not n0 in [1, n1]:
            msg = f"\nERROR: fixed images should be 1 or {n1} files.\n"
            msg += f"{n0} has been given\n\n"
            self.terminal.write(msg)
            return

        if self.inv_reg and n0 != n1:
            msg = f"\nERROR: 'INV' can be activated only when processing N-to-N files'.\n"
            self.terminal.write(msg)
            return

        self.terminal.write("\n")

        self.set_dirname_res(dirname_res=dirname_res)

        fnames_fixed = self.fnames_tot[0]
        fnames_moving = self.fnames_tot[1]
        for i, fname_moving in enumerate(fnames_moving):
            fname_fixed = fnames_fixed[0] if n0 == 1 else fnames_fixed[i]
            names = [Path(fname_fixed).name, Path(fname_moving).name]
            self.terminal.write(f"{i + 1}/{n1} {names[0]} - {names[1]}:\n")

            try:
                self.load_image(0, fname=fname_fixed)
                self.load_image(1, fname=fname_moving)
                if not self.fixed_reg:
                    self.registration_calc()
                imgs = self.registration_apply()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    for k in range(2):
                        iio.imwrite(self.dirname_res[k] / names[k],
                                    imgs[k].astype(self.dtypes[k]))

            except:
                self.terminal.write("FAILED\n")

        self.terminal.write("\n")

    def save_params(self, fname_json=None):
        """ Save parameters in a .json file """
        if fname_json is None:
            fname_json = filedialog.asksaveasfilename(defaultextension='.json')
            if fname_json is None:
                return

        params = {}
        for key in KEYS:
            params.update({key: eval(f"self.{key}")})

        with open(fname_json, 'w', encoding='utf-8') as fid:
            json.dump(params, fid, ensure_ascii=False, indent=4)

    @staticmethod
    def reload_params(fname_json=None, obj=None):
        """ Reload parameters from a .json file and
            Return an ImagesAlign() object"""

        if fname_json is None:
            fname_json = filedialog.askopenfilename(defaultextension='.json')

        if not os.path.isfile(fname_json):
            raise IOError(f"{fname_json} is not a file")

        if obj is not None:
            assert isinstance(obj, ImagesAlign)

        with open(fname_json, 'r', encoding='utf-8') as fid:
            params = json.load(fid)

        imgalign = obj or ImagesAlign()
        for key, value in params.items():
            setattr(imgalign, key, value)
        return imgalign

    def plot_all(self, ax=None):
        """ Plot all the axis """
        if ax is not None:
            self.ax = ax

        for k in range(len(self.ax)):
            self.plot_k(k)

    def plot_k(self, k):
        """ Plot the k-th axis """
        self.ax[k].clear()

        if k in [0, 1]:
            self.plot_fixed_or_moving_image(k)

        elif k == 2:
            self.plot_juxtaposed_images()

        elif k == 3:
            self.plot_combined_images()

        else:
            raise IOError

        self.ax[k].autoscale(tight=True)

    def plot_fixed_or_moving_image(self, k):
        """ Plot the fixed (k=0) or the moving (k=1) image """

        if self.imgs[k] is None:
            return

        self.ax[k].set_title(['Fixed image', 'Moving image'][k])
        extent = [0, self.imgs[k].shape[1], 0, self.imgs[k].shape[0]]

        if self.binarized:
            img = np.zeros_like(self.imgs_bin[k], dtype=int)
            img[self.imgs_bin[k]] = 2 * k - 1
            img = rescaling(img, self.rfactors_plotting[k])
            self.ax[k].imshow(img, cmap=CMAP_BINARIZED, vmin=-1, vmax=1,
                              extent=extent)
        else:
            img = self.imgs[k].copy()
            img = rescaling(img, self.rfactors_plotting[k])
            self.ax[k].imshow(img, cmap='gray', extent=extent)

        if self.rois[k] is not None:
            xmin, xmax, ymin, ymax = self.rois[k]
            width, height = xmax - xmin, ymax - ymin
            self.ax[k].add_patch(Rectangle((xmin, ymin), width, height,
                                           ec='y', fc='none'))

    def plot_juxtaposed_images(self, nmax=10):
        """ Plot the juxtaposed images """

        self.ax[2].set_title("Juxtaposed images")

        imgs = [self.ax[k].get_images() for k in range(2)]
        if len(imgs[0]) == 0 or len(imgs[1]) == 0:
            return

        alignment = self.juxt_alignment
        rfacs = self.rfactors_plotting

        arrs = []
        for k in range(2):
            arr = imgs[k][0].get_array()
            if self.rois[k] is not None:
                roi = (np.asarray(self.rois[k]) * rfacs[k]).astype(int)
                arr = cropping(arr, roi, verbosity=False)
            arrs.append(arr)

        if not self.binarized:
            arrs = [image_normalization(arr) for arr in arrs]
        arrs = imgs_conversion(arrs)
        img, offset = concatenate_images(arrs[0], arrs[1], alignment=alignment)
        extent = [0, img.shape[1], 0, img.shape[0]]

        if self.binarized:
            self.ax[2].imshow(img, cmap=CMAP_BINARIZED, vmin=-1, vmax=1,
                              extent=extent)
        else:
            self.ax[2].imshow(img, cmap='gray', extent=extent)

        x0, y0 = self.rois[0][::2] if self.rois[0] is not None else (0, 0)
        x1, y1 = self.rois[1][::2] if self.rois[1] is not None else (0, 0)

        # draw lines related to 'SIFT' or 'User-Driven' registration mode
        for i, (src, dst) in enumerate(zip(self.points[0][:nmax], self.points[1][:nmax])):
            x = [(src[0] - x0) * rfacs[0], (dst[0] - x1) * rfacs[1] + offset[0]]
            y = [(src[1] - y0) * rfacs[0], (dst[1] - y1) * rfacs[1] + offset[1]]
            self.ax[2].plot(x, y, color=COLORS[i])

    def plot_combined_images(self):
        """ Plot the combined images """

        self.ax[3].set_title("Combined images")

        if self.imgs[0] is None or self.imgs[1] is None:
            return

        rfacs = self.rfactors_plotting

        if self.binarized:
            k0, k1 = (0, 1) if self.inv_reg else (1, 0)
            imgs = self.crop_and_resize(self.imgs_bin, verbosity=False)
            if self.img_reg_bin is not None:
                imgs[k0] = self.img_reg_bin
            imgs = padding(*imgs)
            img = np.zeros_like(imgs[0], dtype=float)
            img[imgs[1] * ~imgs[0]] = 1
            img[imgs[0] * ~imgs[1]] = -1
            if self.apply_mask and self.mask is not None:
                img[self.mask] = np.nan
            img = rescaling(img, rfacs[k1])
            self.ax[3].imshow(img, cmap=CMAP_BINARIZED, vmin=-1, vmax=1)

        else:
            k0, k1 = (0, 1) if self.inv_reg else (1, 0)
            imgs = self.crop_and_resize(self.imgs, verbosity=False)
            if self.img_reg is not None:
                imgs[k0] = self.img_reg
            imgs = padding(*imgs)
            imgs = [image_normalization(img) for img in imgs]
            imgs = imgs_conversion(imgs)
            img = np.stack([imgs[0], imgs[1]], axis=0)
            if self.apply_mask:
                img = np.mean(img, axis=0)
            else:
                img = np.nanmean(img, axis=0)
            img = rescaling(img, rfacs[k1])
            self.ax[3].imshow(img, cmap='gray')
