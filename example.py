"""
Example
"""
import os
from pathlib import Path
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import rotate
from skimage.io import imsave, imread

from app import App


def moving_image_generation(radius):
    img = shepp_logan_phantom()[::4, ::4]  # low image resolution
    if radius is not None:
        shape = img.shape
        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        mask = (x - 70) ** 2 + (y - 70) ** 2 < radius ** 2
        img[mask] = 1.
    img = rotate(img, 10, center=(40, 60), cval=0)  # rotation
    img = np.pad(img, ((40, 0), (60, 0)))  # padding
    return img


with tempfile.TemporaryDirectory() as tmpdirname:
    tmpdirname = Path(tmpdirname)

    # fixed image (high resolution squared image)
    img1 = shepp_logan_phantom()
    imsave(tmpdirname / 'img1.tif', img1)

    # moving image (low resolution rectangular image)
    img2 = moving_image_generation(radius=4)
    imsave(tmpdirname / 'img2.tif', img2)

    # App definition
    fnames = [tmpdirname / 'img1.tif', tmpdirname / 'img2.tif']
    thresholds = [0.15, 0.15]
    bin_inversions = [False, False]
    mode_auto = False

    app = App(fnames=fnames, thresholds=thresholds,
              bin_inversions=bin_inversions, mode_auto=mode_auto)

    # cropping
    app.h_range_sliders[1].value = (0.40, 0.95)
    app.v_range_sliders[1].value = (0.25, 1.00)
    app.cropping(1)

    # resizing - binarization - registration
    app.resizing()
    app.binarization()
    app.registration_calc(registration_model='StackReg')
    # app.registration_calc(registration_model='SIFT')

    # application to a set of images
    input_dirname = tmpdirname / 'inputs'
    output_dirname = tmpdirname / 'results'
    os.makedirs(input_dirname)
    for k in range(3):
        img2 = moving_image_generation(radius=2 + k)
        imsave(input_dirname / f'img2_{k + 1}.tif', img2)
    app.input_dirpath_widget.value = str(input_dirname)
    app.output_dirpath_widget.value = str(output_dirname)
    app.apply()

    # results visualization

    plt.figure(figsize=(10, 3))
    plt.tight_layout()
    plt.subplot(1, 3, 1)
    plt.title('Moving image')
    plt.imshow(imread(tmpdirname / 'img1.tif'), origin='lower', cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('Fixed image')
    plt.imshow(imread(tmpdirname / 'img2.tif'), origin='lower', cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    plt.imshow(imread(output_dirname / 'img1_2.tif'), origin='lower',
               cmap='gray')

    plt.figure(figsize=(10, 3))
    plt.tight_layout()
    for k in range(3):
        name = f'img1_{k + 1}.tif'
        img1_reg = imread(output_dirname / name)
        plt.subplot(1, 3, k + 1)
        plt.title(name)
        plt.imshow(np.maximum(img1_reg, img2), origin='lower', cmap='gray')

    plt.show()

    app.window.show()
