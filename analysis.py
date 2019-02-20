"""
instance.py (python3)
Zhiang Chen, Feb 2019

Copyright (c) 2019 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU

"""

import numpy as np
import cv2
import os
import gdal
import pickle
import matplotlib.pyplot as plt


class contourAnalysis(object):
    def __init__(self):
        self.instances = []
        self.ds = None
        self.x = None
        self.y = None
        self.x_size = None
        self.y_size = None

    def readTiff(self, f):
        assert os.path.isfile(f)
        self.ds = gdal.Open(f)
        coord = self.ds.GetGeoTransform()
        self.x, self.x_size, _, self.y, _, self.y_size = coord

        self.h,self.w = self.ds.GetRasterBand(1).ReadAsArray().shape

        self.X = self.x + self.x_size * self.w
        self.Y = self.y + self.y_size * self.h

        R = self.ds.GetRasterBand(1).ReadAsArray()
        G = self.ds.GetRasterBand(2).ReadAsArray()
        B = self.ds.GetRasterBand(3).ReadAsArray()
        dim = R.shape
        resize = (dim[0], dim[1], 1)
        self.tif = np.concatenate((R.reshape(resize), G.reshape(resize), B.reshape(resize)), axis=2)

        print("Tiff coordinates (x,y):")
        print("Upper Left: ", self.x, self.y)
        print("Lower Right: ", self.X, self.Y)


    def readInstances(self, f, mode="pickle"):
        assert os.path.isfile(f)
        if mode == "pickle":
            with open(f, 'rb') as pk:
                self.instances = pickle.load(pk)

    def registerArea(self, y1, y2, x1=None, x2=None):
        assert (self.y >= y1) & (y1 > y2) & (y2 > self.Y)
        if (x1 == None) | (x2 == None):
            x1 = self.x
            x2 = self.X
        assert (self.x <= x1) & (x1 <= x2) & (x2 <= self.X)
        area = np.array((y1, x1, y2, x2))
        area = self.__coord2pix(area)
        self.ids = self.__findInstances(area)
        print(len(self.ids))

    def getSizeHist(self, nm=100, threshold=4000, display=True):
        self.sizes = []
        for id in self.ids:
            bb = self.instances[id]['bb']
            mask = self.instances[id]['mask']
            image = self.tif[bb[0]:bb[2], bb[1]:bb[3], :]
            top_left = bb[:2]
            mask = mask - top_left
            mask = self.__create_bool_mask(mask, image.shape[:2])
            _, contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            areas = [cv2.contourArea(cnt) for cnt in contours]
            size = np.max(areas)

            if size > threshold:
                continue
            self.sizes.append(size)
            """
            if len(areas) > 1: # for visualization purpose
                i = np.argmax(areas)
                image = cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
                plt.imshow(image)
                plt.show()
            """
        if display:
            num_bins = nm
            fig, ax = plt.subplots()
            l = np.sqrt(np.array(self.sizes))
            n, bins, patches = ax.hist(l, num_bins)
            plt.show()


    def __findInstances(self, area):
        """
        find all instances within the area
        :param area: area coordinates
        :return: id list of the instances
        """
        ids = []
        for i,instance in enumerate(self.instances):
            bb = instance['bb']
            # check if the center of the bounding box is in the area
            cy, cx = (bb[0]+bb[2])/2.0, (bb[1]+bb[3])/2.0
            if (area[0] <= cy <= area[2]) & (area[1] <= cx <= area[3]):
                ids.append(i)
        return ids

    def __coord2pix(self, coord):
        """
        convert GPS coordinates to pixel coordinates
        :param coord: GPS coordinates
        :return: pixel coordinates
        """
        y = (coord[0] - self.y)/self.y_size
        x = (coord[1] - self.x)/self.x_size
        Y = (coord[2] - self.y)/self.y_size
        X = (coord[3] - self.x)/self.x_size
        area = np.array((y, x, Y, X)).astype(int)
        return area

    def __create_bool_mask(self, mask, size):
        """
        maybe there are more efficient ways to do this...
        :param mask: mask by index
        :param size: size of image
        :return: bool mask
        """
        mask_ = np.zeros(size)
        for y, x in mask:
            if (y < size[0]) & (x < size[1]):
                mask_[y, x] = 1
        return mask_

    def __getAxes(self, mask):
        pass

    def __getPolygonPoints(self):
        pass


if __name__  ==  "__main__":
    ca = contourAnalysis()
    ca.readTiff("./datasets/C3/C3.tif")
    ca.readInstances("./datasets/C3/registered_instances_v3.pickle")
    ca.registerArea(4145850, 4145800)
    ca.getSizeHist()
    #ca.registerArea(4146294, 4146244)
    #ca.getSizeHist()