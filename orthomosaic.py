"""
orthomosaic.py
Zhiang Chen, Feb 2019

Copyright (c) 2019 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU

read Tiff; read global instances; add global instances to the tiff and create a heatmap; save the new tiff
"""

import os
import gdal
import pickle
import numpy as np
import colorsys
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class orthotiff(object):
    def __init__(self):
        self.instances = []
        self.ds = None
        self.cnt = []

    def readTiff(self, f):
        assert os.path.isfile(f)
        self.ds = gdal.Open(f)

    def readInstances(self, f, mode="pickle"):
        assert os.path.isfile(f)
        if mode == "pickle":
            with open(f, 'rb') as pk:
                self.instances = pickle.load(pk)

    def readRGB(self, r, g, b):
        r = cv2.imread(r, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        g = cv2.imread(g,  cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        b = cv2.imread(b,  cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        dim = r.shape
        resize = (dim[0], dim[1], 1)
        r = r.reshape(resize)
        g = g.reshape(resize)
        b = b.reshape(resize)
        self.tif = np.concatenate((r, g, b), axis=2)

    def mapInstance(self, N=100, got_tif=False):
        if not got_tif:
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
            if i % 10 == 0:
                print(i)
        print(len(self.instances))

    def savePNG(self, f):
        #tif = cv2.flip(self.tif, 0)  # flip vertically. Somehow the image is flipped. 
        tif = self.tif
        cv2.imwrite(f, tif)

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
        x, y = mask[:, 0].min(), mask[:, 1].min()
        x_, y_ = mask[:, 0].max(), mask[:, 1].max()
        bb = np.array((x, y, x_, y_))

        image = self.tif[bb[0]:bb[2], bb[1]:bb[3], :]

        top_left = bb[:2]
        mask = mask - top_left
        mask = self.__create_bool_mask(mask, image.shape[:2])
        cnt = np.count_nonzero(mask)
        color = self.__get_color(cnt)
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
        mask_[0, :] = 1
        mask_[:, 0] = 1
        mask_[-1, :] = 1
        mask_[:, -1] = 1
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
        # hsv = [(i / N, 1, brightness) for i in range(N)]
        hsv = [(i / float(N), 1, brightness) for i in range(N)]
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

    def __get_color(self, cnt):
        if cnt>30000:
            return (1.0, 1.0, 1.0)

        cmap = cm.get_cmap('plasma')
        return cmap(cnt/3000.0)[:3]

if __name__  ==  "__main__":
    import sys
    #sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2
    
    #"""
    orth = orthotiff()
    orth.readRGB('./datasets/C3/B.png', './datasets/C3/G.png', './datasets/C3/R.png')
    orth.readInstances("./registered_instances_c3_rgbd1_refined.pickle")
    print("mapping instances")
    orth.mapInstance(got_tif=True)
    orth.savePNG("./rocks_c3_rgbd1_refined.png")
    """
    orth = orthotiff()
    orth.readRGB('./datasets/C3/B.png', './datasets/C3/G.png', './datasets/C3/R.png')
    orth.readInstances("./datasets/C3/rocks_c3_rgbd1_00.pickle")
    orth.mapInstance(got_tif=True)
    orth.readInstances("./datasets/C3/rocks_c3_rgbd1_01.pickle")
    orth.mapInstance(got_tif=True)
    orth.readInstances("./datasets/C3/rocks_c3_rgbd1_02.pickle")
    orth.mapInstance(got_tif=True)
    orth.readInstances("./datasets/C3/rocks_c3_rgbd1_03.pickle")
    orth.mapInstance(got_tif=True)
    orth.readInstances("./datasets/C3/rocks_c3_rgbd1_04.pickle")
    orth.mapInstance(got_tif=True)
    orth.readInstances("./datasets/C3/rocks_c3_rgbd1_05.pickle")
    orth.mapInstance(got_tif=True)
    orth.readInstances("./datasets/C3/rocks_c3_rgbd1_06.pickle")
    orth.mapInstance(got_tif=True)
    orth.readInstances("./datasets/C3/rocks_c3_rgbd1_07.pickle")
    orth.mapInstance(got_tif=True)
    orth.readInstances("./datasets/C3/rocks_c3_rgbd1_08.pickle")
    orth.mapInstance(got_tif=True)
    orth.savePNG("./rocks_c3_rgbd1_raw.png")
    """

