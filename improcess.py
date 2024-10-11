import cv2 as cv
import numpy as np
from skimage import exposure, morphology
from skimage.filters import threshold_mean, threshold_otsu
from scipy import signal
import matplotlib.pyplot as plt


def show_histogram(image):
    _, axes = plt.subplots(1, 2)
    if image.ndim == 2:
        hist = exposure.histogram(image)
        axes[0].imshow(image, cmap=plt.get_cmap("gray"))
        axes[0].set_title("Image")
        axes[1].plot(hist[0])
        axes[1].set_title("Histogram")
    else:
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[1].set_title("Histogram")
        colors = ["red", "green", "blue"]
        for i, color in enumerate(colors):
            axes[1].plot(exposure.histogram(image[..., i])[0], color=color)
    plt.show()


def return_histogram_path(image):
    if image.ndim == 2:
        hist = exposure.histogram(image)
        plt.plot(hist[0])
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.title("Histogram")
    else:
        plt.title("Histogram")
        colors = ["red", "green", "blue"]
        for i, color in enumerate(colors):
            plt.plot(exposure.histogram(image[..., i])[0], color=color)
    plt.savefig("histogram.png")
    plt.close()
    return "histogram.png"


def mean_treshold(image):
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = threshold_mean(image)
    binary = (image > thresh) * 225
    return binary.astype("uint8")


def otsu_treshold(image):
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = threshold_otsu(image)
    binary = (image > thresh) * 255
    return binary.astype("uint8")


def dilatation(image):
    bin_img = binarize(image) / 255
    dilation = (morphology.binary_dilation(image=bin_img)) * 255
    return dilation.astype("uint8")


def erosion(image):
    bin_img = binarize(image) / 255
    erosion = (morphology.erosion(image=bin_img)) * 255
    return erosion.astype("uint8")


def resize(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Fonction effectuant le changement d'échelle de l'image `image` selon le facteur `scale`
        en utilisant l'interpolation linéaire.

    Paramètre(s) d'entrée

    image : ndarray
        Image (niveaux de gris) d'un type reconnu par Python.
    scale : float
        Paramètre de changement d'échelle. Un nombre réel strictement positif

    Paramètre(s) de sortie
    ----------------------
    im_resized : ndarray
        Image interpolé àa la nouvelle échelle, de même type que `image`
    """

    new_height = int(image.shape[1] * scale)
    new_width = int(image.shape[0] * scale)
    new_shape = (new_height, new_width)

    inter = cv.INTER_AREA if scale <= 1 else cv.INTER_LANCZOS4
    im_resized = cv.resize(image, new_shape, interpolation=inter)

    return im_resized.astype("uint8")


def negative(image) -> np.ndarray:
    neg_image = 255 - image
    return neg_image.astype("uint8")


def gray_scale(image):
    if image.ndim == 3:
        return cv.cvtColor(image, cv.COLOR_RGB2GRAY).astype("uint8")
    return image


def binarize(image) -> np.ndarray:
    gray_img = gray_scale(image)
    bin_image = np.where(gray_img > 128, 255, 0)
    return bin_image.astype("uint8")


def log_trans(image) -> np.ndarray:
    image = image.astype(float)
    c = 255 / np.log(1 + np.max(image))
    log_img = c * np.log(1 + image)
    log_img = np.clip(log_img, 0, 255)
    return log_img.astype("uint8")


def exp_trans(image) -> np.ndarray:
    image = image.astype(np.float32)

    normalized_img = image / 255.0

    exp_img = np.exp(normalized_img) - 1

    exp_img = 255 * exp_img / np.max(exp_img)

    return exp_img.astype("uint8")


def gamma_trans(image, gamma):
    gamma *= 5
    gamma = 5 - gamma
    gamma_img = image / 255
    gamma_img = np.power(gamma_img, gamma) * 225
    return gamma_img.astype("uint8")


def equalize(image):
    return exposure.equalize_hist(image).astype("uint8")


def gaussian_blur(image):
    return cv.GaussianBlur(image, ksize=(5, 5), sigmaX=1.0).astype("uint8")


def rotate(img, angle):
    (height, width) = img.shape[:2]

    rotPoint = (width // 2, height // 2)
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(img, rotMat, dimensions)


def find_contours(image):
    thresh = binarize(image)
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    blank = np.zeros(image.shape, dtype="uint8")
    contour_img = cv.drawContours(blank, contours, -1, (255, 255, 255), 2)
    return contour_img


def find_edges(image):
    gray_image = gray_scale(image)
    return cv.Canny(gray_image, 125, 175).astype("uint8")


if __name__ == "__main__":
    image = cv.imread("profile.jpeg")
    # show_histogram(image)
    cv.imshow("normal", image)
    cv.imshow(
        "Gamma",
        erosion(image),
    )
    cv.waitKey(0)
