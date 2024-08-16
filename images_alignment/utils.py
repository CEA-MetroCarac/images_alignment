"""
Application for images registration
"""
import sys
import numpy as np
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import AffineTransform
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac
from skimage.util import img_as_float


class Terminal:
    """ Class to 'write' in the console """

    def write(self, message):
        """ Write message in the console """
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


def edges_trend(img):
    """ Estimate if the average value along the image edges is > 0.5 """
    shape = img.shape
    mask = np.ones_like(img, dtype=bool)
    mask[1:-1, 1:-1] = 0
    sum_edges = np.sum(img[mask])
    return (sum_edges / (2 * (shape[0] + shape[1] - 2))) > 0.5


def padding(img1, img2):
    """ Add image padding """
    shape1 = img1.shape
    shape2 = img2.shape

    hmax = max(img1.shape[0], img2.shape[0])
    wmax = max(img1.shape[1], img2.shape[1])

    img1_pad = np.pad(img1, ((0, hmax - shape1[0]), (0, wmax - shape1[1])))
    img2_pad = np.pad(img2, ((0, hmax - shape2[0]), (0, wmax - shape2[1])))

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
                                cross_check=True, max_ratio=0.8)

    src = keypoints[0][matches[:, 0]][:, ::-1]
    dst = keypoints[1][matches[:, 1]][:, ::-1]
    tmat = ransac((src, dst), model_class,
                  min_samples=4, residual_threshold=2)[0].params

    return tmat, src, dst


def concatenate_images(image1, image2, alignment):
    """
    concatenate image1 and image2

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Pairs and image are drawn in this ax.
    image1 : (N, M [, 3]) array
        First grayscale or color image.
    image2 : (N, M [, 3]) array
        Second grayscale or color image.
    points1 : (n, 2) array
        Points coordinates as ``(row, col)`` related to image1.
    points2 : (n, 2) array
        Points coordinates as ``(row, col)`` related to image2.
    alignment : {'horizontal', 'vertical'}, optional
        Whether to show images side by side, ``'horizontal'``, or one above
        the other, ``'vertical'``.
    """
    image1 = img_as_float(image1)
    image2 = img_as_float(image2)

    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1[:image1.shape[0], :image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2[:image2.shape[0], :image2.shape[1]] = image2
        image2 = new_image2

    offset = np.array(image1.shape)
    if alignment == 'horizontal':
        image = np.concatenate([image1, image2], axis=1)
        offset[0] = 0
    elif alignment == 'vertical':
        image = np.concatenate([image1, image2], axis=0)
        offset[1] = 0

    return image, offset


def plot_pairs(ax, image1, image2, points1, points2, alignment='horizontal'):
    """
    Plot pairs

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Pairs and image are drawn in this ax.
    image1 : (N, M [, 3]) array
        First grayscale or color image.
    image2 : (N, M [, 3]) array
        Second grayscale or color image.
    points1 : (n, 2) array
        Points coordinates as ``(row, col)`` related to image1.
    points2 : (n, 2) array
        Points coordinates as ``(row, col)`` related to image2.
    alignment : {'horizontal', 'vertical'}, optional
        Whether to show images side by side, ``'horizontal'``, or one above
        the other, ``'vertical'``.
    """
    image, offset = concatenate_images(image1, image2, alignment)

    ax.imshow(image, cmap='gray')
    ax.axis((0, image1.shape[1] + offset[1], image1.shape[0] + offset[0], 0))

    if points1 is not None:
        np.random.seed(0)
        rng = np.random.default_rng()

        for point1, point2 in zip(points1, points2):
            color = rng.random(3)
            ax.plot((point1[1], point2[1] + offset[1]),
                    (point1[0], point2[0] + offset[0]), '-', color=color)
