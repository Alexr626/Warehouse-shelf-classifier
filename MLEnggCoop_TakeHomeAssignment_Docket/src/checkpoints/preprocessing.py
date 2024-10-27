import numpy as np
import training_data
import pandas as pd
import os
from PIL import Image
from itertools import combinations
import cv2 as cv
import constants

class Augmenter:
    """Class with methods to apply individual augmentations."""

    @staticmethod
    def normalization(img):
        return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)

    @staticmethod
    def blurring(img):
        return cv.GaussianBlur(img, (5, 5), 1)

    @staticmethod
    def hflip(img):
        return cv.flip(img, 1)

    @staticmethod
    def rotate_90(img):
        return cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

    @staticmethod
    def rotate_180(img):
        return cv.rotate(img, cv.ROTATE_180)

def get_np_array_from_path(img_path):
    return cv.imread(img_path)

def apply_combinations(img, transformation_set, transformation_types):
    """
    Applies a combination of transformations to an image.
    """
    transformed_img = img.copy()
    for transformation in transformation_set:
        transformed_img = transformation_types[transformation](transformed_img)
    return transformed_img

def generate_augmented_samples(image_paths, transformation_types, transformations):
    """
    Creates augmented sample images for viewing and saves them to training_data/augmented_samples.
    """
    save_dir = os.path.join(constants.ROOT_DIR, "training_data", "augmented_samples")
    os.makedirs(save_dir, exist_ok=True)

    for img_path in image_paths:
        img_name = os.path.basename(os.path.splitext(img_path)[0])
        img_subdir = os.path.join(save_dir, img_name)
        os.makedirs(img_subdir, exist_ok=True)

        original_img = get_np_array_from_path(img_path)

        for i, transformation_set in enumerate(transformations):
            transformed_img = apply_combinations(original_img, transformation_set, transformation_types)
            transform_str = "_".join(transformation_set)
            save_path = os.path.join(img_subdir, f"{img_name}_aug_{i}_{transform_str}.png")

            cv.imwrite(save_path, transformed_img)

def generate_augmented_training_data(image_paths, transformation_types, transformations):
    """
    Creates augmented training data and saves it as numpy arrays to training_data/preprocessed_arrays.
    """
    save_dir = os.path.join(constants.ROOT_DIR, "training_data", "preprocessed_arrays")
    os.makedirs(save_dir, exist_ok=True)

    for img_path in image_paths:
        img_name = os.path.basename(os.path.splitext(img_path)[0])
        original_img = get_np_array_from_path(img_path)

        for i, transformation_set in enumerate(transformations):
            transformed_img = apply_combinations(original_img, transformation_set, transformation_types)
            transform_str = "_".join(transformation_set)
            save_path = os.path.join(save_dir, f"{img_name}_aug_{i}_{transform_str}.npy")
            np.save(save_path, transformed_img)

def main():
    augmenter = Augmenter()
    labels = pd.read_csv(os.path.join(constants.ROOT_DIR, "training_data", "labels.csv"))

    # Generate list of image paths for each class
    filled_sample_img_names = (labels["image_path"][labels['class'] == "filled"].sample(5)
                               .apply(lambda img_name: os.path.join(constants.ROOT_DIR, "training_data", img_name)).tolist())
    empty_sample_img_names = (labels["image_path"][labels['class'] == "empty"].sample(5)
                              .apply(lambda img_name: os.path.join(constants.ROOT_DIR, "training_data", img_name)).tolist())
    all_image_names = (labels["image_path"]
                       .apply(lambda img_name: os.path.join(constants.ROOT_DIR, "training_data", img_name)).tolist())

    # Combine lists for final training image paths
    sample_image_names = filled_sample_img_names + empty_sample_img_names

    # Define transformations and combinations
    transformation_types = {
        'normalization': augmenter.normalization,
        'blurring': augmenter.blurring,
        'hflip': augmenter.hflip,
        'rotate_90': augmenter.rotate_90,
        'rotate_180': augmenter.rotate_180,
    }

    transformations = [
        'normalization',
        ('normalization', 'blurring'),
        ('normalization', 'hflip'),
        ('normalization', 'rotate_90'),
        ('normalization', 'rotate_180'),
        ('normalization', 'hflip', 'blurring'),
        ('normalization', 'rotate_90', 'blurring'),
        ('normalization', 'rotate_180', 'blurring')
    ]

    # Generate augmented samples for viewing
    generate_augmented_samples(sample_image_names, transformation_types, transformations)

    # Generate augmented training data
    generate_augmented_training_data(all_image_names, transformation_types, transformations)

    print("Augmentation and preprocessing completed successfully.")

if __name__ == "__main__":
    main()