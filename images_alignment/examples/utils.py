"""
Utilities functions dedicated to the examples
"""
from pathlib import Path
import tempfile
from collections import namedtuple
import numpy as np
from skimage import data
from skimage.transform import AffineTransform, warp
from skimage.io import imsave
from skimage import img_as_ubyte
from scipy.ndimage import rotate

from images_alignment.utils import gray_conversion

ROIS = {'astronaut': [[130, 320, 310, 500], [5, 70, 130, 210]],
        'camera': [[130, 330, 290, 460], [5, 80, 130, 210]],
        'shepp_logan_phantom': [None, None]}


class UserTempDirectory:
    """ Class to call user temp via the 'with' statement """

    def __enter__(self):
        return tempfile.gettempdir()

    def __exit__(self, exc, value, tb):
        pass


def moving_image_generation(img0, rotation=0.5):
    """Low resolution image generation with an additional rectangular pattern"""
    img = img0.copy()
    img = img[::2, ::2]  # low image resolution
    tform = AffineTransform(scale=(1.5, 0.8),
                            rotation=rotation,
                            translation=(-50, -100))
    img = warp(img, tform)
    img2 = warp(np.ones((img.shape[0], img.shape[1])), tform)
    imin, imax, jmin, jmax = find_max_inner_rectangle(img2, value=1)
    img = img[imin:imax, jmin:jmax]
    if img0.dtype == np.uint8:
        img = img_as_ubyte(img)

    return img


def images_generation(dirname, img_name='astronaut', nimg=1):
    """ Generate the set of images to handle """
    dirname = Path(dirname)

    # fixed image
    img1 = eval(f"data.{img_name}()")
    fname_fixed = dirname / 'img1.tif'
    imsave(fname_fixed, img1)
    fnames_fixed = [fname_fixed]

    # moving image(s)
    fnames_moving = []
    if nimg == 1:
        img2 = moving_image_generation(img1, rotation=0.5)
        fname_moving = dirname / 'img2.tif'
        fnames_moving.append(fname_moving)
        imsave(fname_moving, img2)
    else:
        for k in range(nimg):
            if img_name == 'shepp_logan_phantom':
                img2 = rotate(img1, angle=-(20 + 20 * k), reshape=False)
                img2 = np.pad(img2, [[200, 50], [200, 50]])
                img2 = img2[::4, ::4]
            else:
                img2 = moving_image_generation(img1, rotation=0.5 + 0.1 * k)
            fname_moving = dirname / f'img2_{k + 1}.tif'
            fnames_moving.append(fname_moving)
            imsave(fname_moving, img2)

    return fnames_fixed, fnames_moving


def find_max_inner_rectangle(arr, value=0):
    """
    Returns coordinates of the largest rectangle containing the 'value'.
    From : https://stackoverflow.com/questions/2478447

    Parameters
    ----------
    arr: numpy.ndarray((m, n), dtype=int)
        2D array to work with
    value: int, optional
        Reference value associated to the area of the largest rectangle

    Returns
    -------
    imin, imax, jmin, jmax: ints
        indices associated to the largest rectangle
    """
    Info = namedtuple('Info', 'start height')

    def rect_max_size(histogram):
        stack = []
        top = lambda: stack[-1]
        max_size = (0, 0, 0)  # height, width and start position of the max rect
        pos = 0  # current position in the histogram
        for pos, height in enumerate(histogram):
            start = pos  # position where rectangle starts
            while True:
                if not stack or height > top().height:
                    stack.append(Info(start, height))  # push
                elif stack and height < top().height:
                    tmp = (top().height, pos - top().start, top().start)
                    max_size = max(max_size, tmp, key=area)
                    start, _ = stack.pop()
                    continue
                break  # height == top().height goes here

        pos += 1
        for start, height in stack:
            max_size = max(max_size, (height, (pos - start), start), key=area)

        return max_size

    def area(size):
        return size[0] * size[1]

    iterator = iter(arr)
    hist = [(el == value) for el in next(iterator, [])]
    max_rect = rect_max_size(hist) + (0,)
    for irow, row in enumerate(iterator):
        hist = [(1 + h) if el == value else 0 for h, el in zip(hist, row)]
        max_rect = max(max_rect, rect_max_size(hist) + (irow + 1,), key=area)

    imax = int(max_rect[3] + 1)
    imin = int(imax - max_rect[0])
    jmin = int(max_rect[2])
    jmax = int(jmin + max_rect[1])

    return imin, imax, jmin, jmax
