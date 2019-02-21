"""
orthomosaic.py
Zhiang Chen, Feb 2019

Copyright (c) 2019 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU

read Tiff; read global instances; add global instances to the tiff and create a heatmap; save the new tiff
"""

import os
import gdal
import cv2
import pickle
import numpy as np
import colorsys
import random
import matplotlib.pyplot as plt

class orthotiff(object):
    def __init__(self):
        self.instances = []
        self.ds = None

    def readTiff(self, f):
        assert os.path.isfile(f)
        self.ds = gdal.Open(f)

    def readInstances(self, f, mode="pickle"):
        assert os.path.isfile(f)
        if mode == "pickle":
            with open(f, 'rb') as pk:
                self.instances = pickle.load(pk)

    def mapInstance(self, N=100):
        R = self.ds.GetRasterBand(1).ReadAsArray()
        G = self.ds.GetRasterBand(2).ReadAsArray()
        B = self.ds.GetRasterBand(3).ReadAsArray()
        dim = R.shape
        resize = (dim[0], dim[1], 1)
        self.tif = np.concatenate((R.reshape(resize), G.reshape(resize), B.reshape(resize)), axis=2)

        colors = self.__random_colors(N)

        for i,instance in enumerate(self.instances):
            color_id = np.random.randint(N)
            self.__add_instance(instance, colors[color_id])
            print(i)


    def saveTiff(self, f):
        driver = gdal.GetDriverByName('GTiff')
        y,x,c = self.tif.shape

        dataset = driver.Create(
            f,
            x,
            y,
            c,
            gdal.GDT_Byte)

        md = self.ds.GetMetadata()
        gt = self.ds.GetGeoTransform()
        pj = self.ds.GetProjection()
        dataset.SetGeoTransform(gt)
        dataset.SetMetadata(md)
        dataset.SetProjection(pj)
        dataset.GetRasterBand(1).WriteArray(self.tif[:, :, 0])
        dataset.GetRasterBand(2).WriteArray(self.tif[:, :, 1])
        dataset.GetRasterBand(3).WriteArray(self.tif[:, :, 2])
        dataset.FlushCache()

    def __add_instance(self, instance, color):
        bb = instance['bb']
        mask = instance['mask']
        image = self.tif[bb[0]:bb[2], bb[1]:bb[3], :]

        top_left = bb[:2]
        mask = mask - top_left
        mask = self.__create_bool_mask(mask, image.shape[:2])
        image = self.__apply_mask(image, mask, color)

        self.tif[bb[0]:bb[2], bb[1]:bb[3], :] = image



    def __create_bool_mask(self, mask, size):
        """
        maybe there are more efficient ways to do this...
        :param mask: mask by index
        :param size: size of image
        :return: bool mask
        """
        mask_ = np.zeros(size)
        mask_[0,:] = 1
        mask_[:,0] = 1
        mask_[-1,:] = 1
        mask_[:,-1] = 1
        for y,x in mask:
            if (y<size[0]) & (x<size[1]):
                mask_[y,x] = 1
        return mask_

    def __random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def __apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """

        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image


if __name__  ==  "__main__":
    orth = orthotiff()
    orth.readTiff("./datasets/C3/C3.tif")
    #orth.readInstances("./datasets/C3/registered_instances_v3.pickle")
    orth.readInstances("./talk.pickle")
    print("mapping instances")
    orth.mapInstance()
    orth.saveTiff("./talk.tif")
