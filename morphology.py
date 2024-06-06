
import cv2
import numpy as np
from skimage import morphology
import logging

__authors__ = ("BAGUIAN Harouna", "HYUNH Ashley")
__contact__ = ("baguian.harouna7231@gmail.com",
               "harouna.baguian7231@gmail.com")
__copyright__ = "MIT"
__date__ = "2024-04-06"
__version__ = "1.0.0"


class Morphology:

    def __init__(self):
        pass

    def get_scale(self, image):
        """
        The function `get_scale` processes an image to calculate 
        a unit measurement based on a scale and extracts objects from the 
        image.

        Args:
            image: image containing 

        Return: The `get_scale` function returns two values: `unit_mesure` and 
        `Obj`. `unit_mesure` is a calculated unit measurement based on the 
        input image, and `Obj` is a list containing objects extracted from 
        the image.
        """
        scale = []
        Obj = []

        try:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(
                image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            thresholds = cv2.bitwise_not(th)
            contours, _ = cv2.findContours(
                thresholds, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for i in range(0, len(contours)):
                mask_BB_i = np.zeros(
                    (len(thresholds), len(thresholds[0])), np.uint8)
                x, y, w, h = cv2.boundingRect(contours[i])
                cv2.drawContours(mask_BB_i, contours, i, (255, 255, 255), -1)
                BB_i = cv2.bitwise_and(thresholds, thresholds, mask=mask_BB_i)
                if h > 15 and w > 15:
                    BB_i = BB_i[y:y+h, x:x+w]
                    Obj.append(BB_i)

                if h < 7 and w > 15:
                    BB_i = BB_i[y:y+h, x:x+w]
                    scale.append(BB_i)

        except Exception as e:
            logging.error(f"error: {e}")
        Ech_area = np.sum(scale[0] == 255)-10

        unit_mesure = 100/Ech_area
        return unit_mesure, Obj

    def get_obj(self, image):
        """
        The function `get_scale` processes an image to calculate 
        a unit measurement based on a scale and extracts objects from the 
        image.

        Args:
            image: image containing 

        Return: The `get_scale` function returns two values: `unit_mesure` and 
        `Obj`. `unit_mesure` is a calculated unit measurement based on the 
        input image, and `Obj` is a list containing objects extracted from 
        the image.
        """
        scale = []
        Obj = []

        try:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(
                image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            thresholds = cv2.bitwise_not(th)
            contours, _ = cv2.findContours(
                thresholds, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for i in range(0, len(contours)):
                mask_BB_i = np.zeros(
                    (len(thresholds), len(thresholds[0])), np.uint8)
                x, y, w, h = cv2.boundingRect(contours[i])
                cv2.drawContours(mask_BB_i, contours, i, (255, 255, 255), -1)
                BB_i = cv2.bitwise_and(thresholds, thresholds, mask=mask_BB_i)
                if h > 15 and w > 15:
                    BB_i = BB_i[y:y+h, x:x+w]
                    Obj.append(BB_i)

                if h < 7 and w > 15:
                    BB_i = BB_i[y:y+h, x:x+w]
                    scale.append(BB_i)

        except Exception as e:
            logging.error(f"error: {e}")
        
        return Obj

    def skeletonization(self, image, obj):
        """
        The function skeletonization takes an image, identifies the largest 
        object in the image, performs morphological operations to 
        extract its skeleton, and returns the skeletonized image.

        Args: 
            image: The `skeletonization` function takes an image as input and 
            performs skeletonization on the largest object in the image. 
            The function first extracts the largest object from the image 
            using the `get_scale` method. It then calculates the area of each 
            object and selects the object with the largest area.

        Return: the function return skeleton of target in the input image.
        """
        Obj = obj
        area = 0
        for i in range(0, len(Obj)):
            contours, _ = cv2.findContours(
                Obj[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            temp_area = cv2.contourArea(cnt)
            if (area < temp_area):
                area = temp_area
                animal_image = Obj[i]

        kernel = np.ones((12, 12), np.uint8)
        eroded_image = cv2.morphologyEx(animal_image, cv2.MORPH_CLOSE, kernel)
        skeleton = morphology.skeletonize(eroded_image)
        skeleton = skeleton.astype(np.uint8)
        return skeleton

    def calculate_length(self, image, scale, obj):
        """
        The function calculates the length of a skeletonized image based on 
        the distances between skeleton points and a scale factor.

        Args:
            image: The `calculate_length` function takes an image as input and 
            performs skeletonization on the image to extract the skeleton. 
            It then calculates the total length of the skeleton by summing 
            the distances between consecutive points in the skeleton. 
            The scale of the image is also taken into account to calculate 
            the length.
        Return: The `calculate_length` function is returning the calculated 
        length of the skeleton in the input image after processing it through 
        skeletonization and scaling.
        """
        skeleton = self.skeletonization(image, obj)
        # scale, _ = self.get_scale(image)
        logging.warning("calculate length start")

        try:
            skeleton_coords = np.argwhere(skeleton > 0)

            total_distance = 0
            for i in range(len(skeleton_coords) - 1):
                point1 = skeleton_coords[i]
                point2 = skeleton_coords[i+1]
                distance = np.linalg.norm(point1 - point2)
                total_distance += distance

            length = total_distance * scale
            return length
        except Exception as e:
            logging.error(f"error: {e}")

    def calculate_width(self, image, scale):
        """
        The function calculates the width of an image.

        :Args:
            image: The `calculate_width` method takes an image as input and 
            performs a series of image processing operations to calculate 
            the width of a specific feature in the image.

        Return: The function `calculate_width` returns the calculated width 
        of the image after performing certain image processing operations.
        """

        try:
            # scale, _ = self.get_scale(image)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(
                image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            image = cv2.bitwise_not(th)
            skeleton = np.zeros(image.shape, np.uint8)
            image_copy = image.copy()
            iterations = 0
            while True:
                temp = cv2.erode(image_copy, np.ones(
                    (3, 3), np.uint8), iterations=1)
                temp = cv2.dilate(temp, np.ones(
                    (3, 3), np.uint8), iterations=1)
                temp = cv2.subtract(image_copy, temp)
                skeleton = cv2.bitwise_or(skeleton, temp)
                image_copy = cv2.erode(image_copy, np.ones(
                    (3, 3), np.uint8), iterations=1)
                iterations += 1
                if cv2.countNonZero(image_copy) == 0:
                    break
            nb_pixel = 2*iterations + 1
            largeur_reel = nb_pixel*scale
            return largeur_reel
        except Exception as e:
            logging.error(f"error: {e}")
