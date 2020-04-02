import numpy as np
import pandas as pd

from skimage import color, feature, io, util


def extract_single_image_feature(image, label):
    """
    This function aims to extract features of a single images
    The features consist of:
        - average red in the image
        - average green in the image
        - average blue in the image
        - average saturation in the image
        - contrast of the image
        - angular second moment (ASM) of the image
        - homogeneity of the image
        - image label
    """

    RGB = color.gray2rgb(image)
    RGB = RGB[:, :, :3]

    R = RGB[:, :, 0]
    G = RGB[:, :, 1]
    B = RGB[:, :, 2]
    avg_r = np.average(R.flatten())
    avg_g = np.average(G.flatten())
    avg_b = np.average(B.flatten())

    HSV = color.rgb2hsv(RGB)
    S = HSV[:, :, 1]
    avg_s = np.average(S.flatten())

    BW = color.rgb2gray(RGB)
    BW = util.img_as_ubyte(BW)
    cov_matrix = feature.greycomatrix(
        BW,
        [1],
        [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        normed=True,
        symmetric=True,
    )

    contrast = feature.greycoprops(cov_matrix, "contrast")[0, 0]
    ASM = feature.greycoprops(cov_matrix, "ASM")[0, 0]
    homogeneity = feature.greycoprops(cov_matrix, "homogeneity")[0, 0]

    return [avg_r, avg_g, avg_b, avg_s, contrast, ASM, homogeneity, label]


def extract_images_features(image_paths, labels):
    """
    This function aims to extract images features given
        their paths and labels
    image_paths: list of image path in directory
    labels: list of label corresponding to each image
    """

    assert len(image_paths) == len(labels)

    data_shape_feature = []
    files_name = []
    dataset_size = len(image_paths)
    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        print("Processing {} / {} images".format(i + 1, dataset_size))

        img = io.imread(img_path)
        data_shape_feature.append([img_path, *extract_single_image_feature(img, label)])

    header = [
        "img_path",
        "avg_r",
        "avg_g",
        "avg_b",
        "avg_s",
        "contrast",
        "ASM",
        "homogeneity",
        "type_label",
    ]

    return pd.DataFrame(data_shape_feature, columns=header)


if __name__ == "__main__":
    pass
