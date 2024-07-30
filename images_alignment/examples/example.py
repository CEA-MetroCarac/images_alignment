"""
Example
"""
from pathlib import Path
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import rotate
from skimage.io import imsave

from images_alignment import ImagesAlign


class UserTempDirectory:
    """ Class to call user temp via the 'with' statement """

    def __enter__(self):
        return tempfile.gettempdir()

    def __exit__(self, exc, value, tb):
        pass


def moving_image_generation(radius):
    """Low resolution image generation with an additional rectangular pattern"""
    img = shepp_logan_phantom()[::4, ::4]  # low image resolution
    if radius is not None:
        shape = img.shape
        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        mask = (x - 70) ** 2 + (y - 70) ** 2 < radius ** 2
        img[mask] = 1.
    img = rotate(img, 10, center=(40, 60), cval=0)  # rotation
    img = np.pad(img, ((40, 0), (60, 0)))  # padding
    return img


def images_generation(dirname):
    """ Generate the set of images to handle """

    # fixed image (high resolution squared image)
    img1 = shepp_logan_phantom()
    fname_fixed = dirname / 'img1.tif'
    imsave(fname_fixed, img1)
    fnames_fixed = [fname_fixed]

    # moving images (low resolution rectangular images)
    img2 = moving_image_generation(radius=4)
    imsave(dirname / 'img2.tif', img2)
    fnames_moving = []
    for k in range(3):
        img2 = moving_image_generation(radius=2 + k)
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
                           bin_inversions=[False, False],
                           mode_auto=False)

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
