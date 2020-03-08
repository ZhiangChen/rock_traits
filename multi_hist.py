#!/usr/bin/env python
"""
strike_hist.py (python3)
Zhiang Chen, Feb 2020

Copyright (c) 2019 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU
"""

import os
import gdal
import pickle
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from math import sqrt



class Multi_Hist(object):
    def __init__(self, instance_path):
        with open(instance_path, 'rb') as pk:
            self.instances = pickle.load(pk)
            print(len(self.instances))

    def readFault(self, path):
        self.fault = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def refineRocks(self):
        instances = []
        i = 0
        for instance in self.instances:
            bbox = instance['bb']
            x = int((bbox[0] + bbox[2])/2)
            y = int((bbox[1] + bbox[3])/2)
            if self.fault[x, y] == 255:
                instances.append(instance)
        print(len(instances))
        return instances

    def plotHist(self, instances, mode, size_max=4000, size_min=10, bin_nm=80):
        rock_sizes = []
        orientations = []
        major_lengths = []
        for rock in instances:
            bb = rock["bb"]
            mask = rock["mask"]
            top_left = bb[:2]
            mask = mask - top_left
            mask_shape = (bb[2] - bb[0], bb[3] - bb[1])
            mask = self.__create_bool_mask(mask, mask_shape)
            # _, contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # cv2.findContours has been changed in Ubuntu 18
            contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            areas = [cv2.contourArea(cnt) for cnt in contours]
            rock_size = np.max(areas)
            if (rock_size > size_max) | (rock_size < size_min):
                continue
            rock_sizes.append(rock_size)

            i = np.argmax(areas)
            cnt = contours[i]
            if cnt.shape[0] < 5:
                continue
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            a = ma / 2
            b = MA / 2
            eccentricity = sqrt(pow(a, 2) - pow(b, 2))
            eccentricity = round(eccentricity / a, 2)
            orientations.append([angle, eccentricity])
            major_lengths.append(2 * a)

        size_step = int((size_max - size_min) / bin_nm)
        size_bins = np.arange(0, size_max, size_step)

        if mode == 'size':
            print(min(rock_sizes))
            plt.hist(rock_sizes, bins= bin_nm)
            plt.show()


    def savePickle(self, instances, path):
        with open(path, 'wb') as pk:
            pickle.dump(instances, pk)


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


if __name__ == '__main__':
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

    mh = Multi_Hist('registered_instances_8.pickle')
    mh.readFault('./datasets/Rock/fault.png')
    rocks = mh.refineRocks()
    #mh.savePickle(rocks, 'registered_instances_fault_6.pickle')
    mh.plotHist(rocks, 'size', size_max=1000)