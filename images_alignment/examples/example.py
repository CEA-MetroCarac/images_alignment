"""
Example
"""
from pathlib import Path
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import rotate
from skimage.io import imsave, imread

from images_alignment.application.app import App


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


def example(dirfunc, show_plots=True):
    """ Example based on 3 duplicated moving images with additional patterns """
    with dirfunc() as tmpdir:
        dirname = Path(tmpdir) / "images_alignement"
        dirname.mkdir(exist_ok=True)

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

        # App definition
        app = App(fnames_fixed=fnames_fixed,
                  fnames_moving=fnames_moving,
                  thresholds=[0.15, 0.15],
                  bin_inversions=[False, False],
                  mode_auto=False)

        # cropping - resizing - binarization - registration
        app.cropping(1, area_percent=[0.40, 0.95, 0.25, 1.00])
        app.resizing()
        app.binarization()
        app.registration(registration_model='StackReg')
        # app.registration(registration_model='SIFT')

        # apply the transformation to the set of images
        dirname_res = dirname / 'results'
        app.apply(dirname_res=dirname_res)

        if show_plots:
            plt.rcParams['image.cmap'] = 'gray'
            plt.rcParams['image.origin'] = 'lower'

            img1 = imread(dirname / 'img1.tif')
            img2 = imread(dirname / 'img2_1.tif')
            img1_res = imread(dirname_res / 'fixed_images' / 'img1.tif')
            img2_res = imread(dirname_res / 'moving_images' / 'img2_1.tif')

            plt.figure(figsize=(10, 3))
            plt.tight_layout()
            plt.subplot(1, 3, 1)
            plt.title('Fixed image')
            plt.imshow(img1)
            plt.subplot(1, 3, 2)
            plt.title('Moving image 1')
            plt.imshow(img2)
            plt.subplot(1, 3, 3)
            plt.title('Overlay 1')
            plt.imshow(0.5 * (img1_res + img2_res))

            plt.figure(figsize=(10, 3))
            plt.tight_layout()
            for k in range(3):
                name = f'img2_{k + 1}.tif'
                img2_res = imread(dirname_res / 'moving_images' / name)
                plt.subplot(1, 3, k + 1)
                plt.title(f'Overlay {k + 1}')
                plt.imshow(0.5 * (img1_res + img2_res))

            plt.show()

        return app


if __name__ == '__main__':
    my_dirfunc = UserTempDirectory  # use the user temp location
    # my_dirfunc = tempfile.TemporaryDirectory  # use a TemporaryDirectory

    my_app = example(my_dirfunc, show_plots=False)
    my_app.window.show()
