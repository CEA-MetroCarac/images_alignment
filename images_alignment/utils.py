"""
Application for images registration
"""
import sys
import numpy as np
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import AffineTransform, resize
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac
from skimage import img_as_ubyte


class Terminal:
    """ Class to 'write' into the console """

    def write(self, message):
        """ Write message into the console """
        sys.stdout.write(message)
        sys.stdout.flush()


def gray_conversion(img):
    """ Convert RGBA or RGB image to gray image """
    if img.ndim == 4:
        img = rgba2rgb(img)
    if img.ndim == 3:
        img = rgb2gray(img)
    return img


def image_normalization(img):
    """ Normalize image in range [0., 1.] """
    vmin, vmax = img.min(), img.max()
    return (img - vmin) / (vmax - vmin)


def resizing(img1, img2):
    """ Resize the images to have similar shape (requested for pyStackReg) """
    if img1.size <= img2.size:
        img1 = resize(img1, img2.shape, preserve_range=True)
    else:
        img2 = resize(img2, img1.shape, preserve_range=True)
    return [img1, img2]


def cropping(img, area):
    """ Return cropped image according to the given area """
    if area is None:
        return img
    else:
        assert np.asarray(area).dtype == int
        shape = img.shape
        xmin, xmax, ymin, ymax = area
        imin, imax = shape[0] - ymax, shape[0] - ymin
        jmin, jmax = xmin, xmax
        return img[imin:imax, jmin:jmax]


def padding(img1, img2):
    """ Add image padding """
    shape1 = img1.shape
    shape2 = img2.shape

    hmax = max(shape1[0], shape2[0])
    wmax = max(shape1[1], shape2[1])

    if len(shape1) == 2:
        pad_width1 = ((0, hmax - shape1[0]), (0, wmax - shape1[1]))
        pad_width2 = ((0, hmax - shape2[0]), (0, wmax - shape2[1]))
    else:
        pad_width1 = ((0, hmax - shape1[0]), (0, wmax - shape1[1]), (0, 0))
        pad_width2 = ((0, hmax - shape2[0]), (0, wmax - shape2[1]), (0, 0))

    img1_pad = np.pad(img1, pad_width1)
    img2_pad = np.pad(img2, pad_width2)

    return img1_pad, img2_pad


def sift(img1, img2, model_class=None):
    """
    SIFT feature detection and descriptor extraction

    Parameters
    ----------
    img1, img2: numpy.ndarray((m, n)), numpy.ndarray((p, q))
        The input images
    model_class: Objet
        'model_class' used by `RANSAC
        <https://scikit-image.org/docs/stable/api/skimage.measure.html>`_.
        If None, consider ``AffineTransform`` from skimage.transform.

    Returns
    -------
    tmat: numpy.ndarrays((3, 3))
        The related transformation matrix
    keypoints: list of 2 numpy.ndarray((n, 2)
        Keypoints coordinates as (row, col) related to the 2 input images.
    descriptors: list of 2 numpy.ndarray((n, p)
        Descriptors associated with the keypoints.
    matches: numpy.ndarray((q, 2))
        Indices of corresponding matches returned by
        skimage.feature.match_descriptors.

    """
    if model_class is None:
        model_class = AffineTransform

    sift_ = SIFT()

    keypoints = []
    descriptors = []
    for img in [img1, img2]:
        sift_.detect_and_extract(img)
        keypoints.append(sift_.keypoints)
        descriptors.append(sift_.descriptors)

    matches = match_descriptors(descriptors[0], descriptors[1],
                                cross_check=True, max_ratio=0.9)

    src = np.asarray(keypoints[0][matches[:, 0]]).reshape(-1, 2)[:, ::-1]
    dst = np.asarray(keypoints[1][matches[:, 1]]).reshape(-1, 2)[:, ::-1]
    model, inliers = ransac((src, dst), model_class,
                            min_samples=4, residual_threshold=2)

    tmat = model.params
    points = [src[inliers], dst[inliers]]

    return tmat, points


def concatenate_images(img1, img2, alignment='horizontal'):
    """
    concatenate img1 and img2

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Pairs and image are drawn in this ax.
    img1 : (N, M [, 3]) array
        First grayscale or color image.
    img2 : (N, M [, 3]) array
        Second grayscale or color image.
    alignment : {'horizontal', 'vertical'}, optional
        Whether to show images side by side, ``'horizontal'``, or one above
        the other, ``'vertical'``.
    """
    new_shape1 = list(img1.shape)
    new_shape2 = list(img2.shape)

    if img1.shape[0] < img2.shape[0]:
        new_shape1[0] = img2.shape[0]
    elif img1.shape[0] > img2.shape[0]:
        new_shape2[0] = img1.shape[0]

    if img1.shape[1] < img2.shape[1]:
        new_shape1[1] = img2.shape[1]
    elif img1.shape[1] > img2.shape[1]:
        new_shape2[1] = img1.shape[1]

    if new_shape1 != img1.shape:
        new_img1 = np.zeros(new_shape1, dtype=img1.dtype)
        new_img1[:img1.shape[0], :img1.shape[1]] = img1
        img1 = new_img1

    if new_shape2 != img2.shape:
        new_img2 = np.zeros(new_shape2, dtype=img2.dtype)
        offset_y = new_shape2[0] - img2.shape[0]
        new_img2[offset_y:offset_y + img2.shape[0], :img2.shape[1]] = img2
        img2 = new_img2

    shape = img1.shape
    offset = np.array([shape[1], shape[0]])
    if alignment == 'horizontal':
        image = np.concatenate([img1, img2], axis=1)
        offset[1] = 0
    elif alignment == 'vertical':
        image = np.concatenate([img1, img2], axis=0)
        offset[0] = 0

    return image, offset


def resizing_for_plotting(img, resizing_factor=0.25):
    """ Return image with 'resizing_factor' applied in the 2 dimensions """
    if resizing_factor == 1.:
        return img
    else:
        shape = img.shape
        shape2 = (int(shape[0] * resizing_factor),
                  int(shape[1] * resizing_factor))
        img2 = resize(img, shape2, order=0)
        return img2
