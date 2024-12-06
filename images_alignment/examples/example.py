"""
Examples in python scripting
"""
from pathlib import Path
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import rotate
from skimage.io import imsave, imread

from images_alignment import ImagesAlign
from images_alignment.examples.utils import UserTempDirectory, images_generation
from images_alignment.examples.utils import ROIS
from images_alignment import REG_MODELS


def example(dirname, img_name, registration_model):
    """ Example """

    input_dirname = dirname / 'example'
    input_dirname.mkdir(exist_ok=True)

    fnames_fixed, fnames_moving = images_generation(dirname, img_name)

    imgalign = ImagesAlign(fnames_fixed=fnames_fixed,
                           fnames_moving=fnames_moving,
                           rois=ROIS[img_name])

    plt.close()  # to close the default figure

    fig0, ax0 = plt.subplots(1, 4, figsize=(12, 4))
    fig0.tight_layout()
    fig0.canvas.manager.set_window_title("Original images")
    imgalign.plot_all(ax=ax0)

    imgalign.registration(registration_model=registration_model)

    fig1, ax1 = plt.subplots(1, 4, figsize=(12, 4))
    fig1.tight_layout()
    fig1.canvas.manager.set_window_title("Processed images")
    imgalign.plot_all(ax=ax1)

    fig2, ax2 = plt.subplots(1, 4, figsize=(12, 4))
    fig2.tight_layout()
    fig2.canvas.manager.set_window_title("Processed images (Binarized)")
    imgalign.binarized = True
    imgalign.plot_all(ax=ax2)


def example_series(dirname):
    """ Example based on 3 shepp_logan images with a spot of variable size """
    input_dirname = dirname / 'example_series' / 'inputs'
    output_dirname = dirname / 'example_series' / 'results'
    input_dirname.mkdir(exist_ok=True)
    output_dirname.mkdir(exist_ok=True)

    # fixed image (high resolution squared image)
    img1 = shepp_logan_phantom()  #
    imsave(input_dirname / 'img1.tif', img1)

    # moving images (low resolution rectangular images)
    img2 = shepp_logan_phantom()[::4, ::4]  # low image resolution
    y, x = np.meshgrid(np.arange(img2.shape[0]), np.arange(img2.shape[1]))
    for k in range(3):
        img2_ = img2.copy()
        radius = 2 + k
        mask = (x - 70) ** 2 + (y - 70) ** 2 < radius ** 2
        img2_[mask] = 1.
        img2_ = rotate(img2_, 10, center=(40, 60), cval=0)  # rotation
        img2_ = np.pad(img2_, ((40, 0), (60, 0)))  # padding
        imsave(input_dirname / f'img2_{k + 1}.tif', img2_)

    imgalign = ImagesAlign(fnames_fixed=[input_dirname / 'img1.tif'],
                           fnames_moving=[input_dirname / 'img2_1.tif',
                                          input_dirname / 'img2_2.tif',
                                          input_dirname / 'img2_3.tif'])

    imgalign.registration_model = 'SIFT'
    imgalign.apply_to_all(dirname_res=output_dirname)

    plt.close()  # to close the default figure

    # results visualization
    plt.figure(figsize=(10, 3))
    plt.tight_layout()
    plt.subplot(1, 3, 1)
    plt.title('Fixed image')
    plt.imshow(imread(input_dirname / 'img1.tif'), origin='lower', cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('Moving image')
    plt.imshow(imread(input_dirname / 'img2_1.tif'), origin='lower', cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    plt.imshow(imread(output_dirname / 'moving_images' / 'img2_1.tif'), origin='lower', cmap='gray')

    plt.figure(figsize=(10, 3))
    plt.tight_layout()
    for k in range(3):
        name = f'img2_{k + 1}.tif'
        img2_reg = imread(output_dirname / 'moving_images' / name)
        plt.subplot(1, 3, k + 1)
        plt.title(name)
        plt.imshow(np.maximum(img2_reg, img1), origin='lower', cmap='gray')


if __name__ == '__main__':
    DIRFUNC = UserTempDirectory  # use the user temp location
    # DIRFUNC = tempfile.TemporaryDirectory  # use a TemporaryDirectory
    IMG_NAMES = ['camera', 'astronaut', 'shepp_logan_phantom']

    with DIRFUNC() as tmpdir:
        dirname = Path(tmpdir) / "images_alignement"
        dirname.mkdir(exist_ok=True)

        example(dirname, IMG_NAMES[0], REG_MODELS[1])
        example_series(dirname)

    plt.show()
