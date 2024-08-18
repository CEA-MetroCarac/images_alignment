"""
Example
"""
from pathlib import Path
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import AffineTransform, warp
from skimage.io import imsave

from images_alignment import ImagesAlign


class UserTempDirectory:
    """ Class to call user temp via the 'with' statement """

    def __enter__(self):
        return tempfile.gettempdir()

    def __exit__(self, exc, value, tb):
        pass


def moving_image_generation(img0, radius):
    """Low resolution image generation with an additional rectangular pattern"""
    img = img0.copy()
    img = img[::2, ::2]  # low image resolution
    tform = AffineTransform(scale=(1.5, 0.8),
                            rotation=0.5,
                            translation=(-50, -100))
    img = warp(img, tform)
    return img


def images_generation(dirname):
    """ Generate the set of images to handle """

    # fixed image (high resolution squared image)
    # img1 = data.shepp_logan_phantom()
    # img1 = data.astronaut()
    img1 = data.camera()

    fname_fixed = dirname / 'img1.tif'
    imsave(fname_fixed, img1)
    fnames_fixed = [fname_fixed]

    # moving images (low resolution rectangular images)
    img2 = moving_image_generation(img1, radius=4)
    imsave(dirname / 'img2.tif', img2)
    fnames_moving = []
    for k in range(3):
        img2 = moving_image_generation(img1, radius=2 + k)
        fname_moving = dirname / f'img2_{k + 1}.tif'
        imsave(fname_moving, img2)
        fnames_moving.append(fname_moving)

    return fnames_fixed, fnames_moving


def example(dirname):
    """ Example based on 3 duplicated moving images with additional patterns """

    fnames_fixed, fnames_moving = images_generation(dirname)

    imgalign = ImagesAlign(fnames_fixed=fnames_fixed,
                           fnames_moving=fnames_moving,
                           thresholds=[0.15, 0.15],
                           bin_inversions=[False, False])

    plt.close()  # to close the default figure
    fig0, ax0 = plt.subplots(1, 3, figsize=(12, 4))
    fig0.suptitle("Original images")
    imgalign.plot(ax=ax0, mode="Gray")

    imgalign.cropping(1, area_percent=[0.40, 0.95, 0.25, 1.00])
    imgalign.resizing()
    imgalign.binarization()
    imgalign.registration(registration_model='StackReg')

    # apply the transformation to the set of images
    imgalign.apply_to_all(dirname_res=dirname / 'results')

    fig1, ax1 = plt.subplots(1, 3, figsize=(12, 4))
    fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4))
    fig1.suptitle("Processed images (Gray mode)")
    fig2.suptitle("Processed images (Binarized mode)")
    imgalign.plot(ax=ax1, mode="Gray")
    imgalign.plot(ax=ax2, mode="Binarized")
    plt.show()


if __name__ == '__main__':
    # dirfunc = UserTempDirectory  # use the user temp location
    dirfunc = tempfile.TemporaryDirectory  # use a TemporaryDirectory

    with dirfunc() as tmpdir:
        dirname = Path(tmpdir) / "images_alignement"
        dirname.mkdir(exist_ok=True)

        example(dirname)
