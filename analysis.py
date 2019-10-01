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
from math import *
from matplotlib.collections import PatchCollection
import matplotlib
import scipy.stats as stats


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

        self.h, self.w = self.ds.GetRasterBand(1).ReadAsArray().shape

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

    def read_contour(self, file_path, threshold=10):
        assert os.path.isfile(file_path)
        return cv2.imread(file_path)[:,:,0] > threshold

    def readInstances(self, f, mode="pickle"):
        assert os.path.isfile(f)
        if mode == "pickle":
            with open(f, 'rb') as pk:
                self.instances = pickle.load(pk)

    def refineInstances(self, contour_path):
        assert os.path.isfile(contour_path)

        ct = (self.read_contour(contour_path) * 255).astype(np.uint8)
        ct = cv2.resize(ct, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)

        ct = ct > 10
        refine_instances = []

        for instance in self.instances:
            bb = instance["bb"]
            h = int((bb[0] + bb[2])/2)
            w = int((bb[1] + bb[3])/2)
            if ct[h, w]:
                refine_instances.append(instance)

        self.instances = refine_instances
        print(len(self.instances))
        #with open("../refined_instances.pickle", 'wb') as pk:
        #    pickle.dump(self.instances, pk)




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

    def saveRegisteredInstances(self, f, mode='pickle'):
        registered_instances = []
        for id in self.ids:
            registered_instances.append(self.instances[id])
        if mode == "pickle":
            with open(f, 'wb') as pk:
                pickle.dump(registered_instances, pk)

    def getSizeHist(self, nm=80, threshold=8000, display=True):
        self.sizes = []
        for id in self.ids:
            bb = self.instances[id]['bb']
            mask = self.instances[id]['mask']
            image = self.tif[bb[0]:bb[2], bb[1]:bb[3], :]
            top_left = bb[:2]
            mask = mask - top_left
            mask = self.__create_bool_mask(mask, image.shape[:2])
            #_, contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # one of these two works
            contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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
            l = np.array(self.sizes)
            n, bins, patches = ax.hist(l, num_bins)
            plt.show()

    def getMajorLengthHist(self, nm=80, threshold=8000, display=True):
        """

        :param nm: the number of bins
        :param threshold: the threshold of rock size in pixel
        :param display:
        :return:
        """
        self.major_lengths = []
        for id in self.ids:
            bb = self.instances[id]['bb']
            mask = self.instances[id]['mask']
            image = self.tif[bb[0]:bb[2], bb[1]:bb[3], :]
            top_left = bb[:2]
            mask = mask - top_left
            mask = self.__create_bool_mask(mask, image.shape[:2])
            #_, contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # one of these two works
            contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            areas = [cv2.contourArea(cnt) for cnt in contours]
            size = np.max(areas)

            if size > threshold:
                continue

            i = np.argmax(areas)
            cnt = contours[i]
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            a = ma / 2
            b = MA / 2

            self.major_lengths.append(ma)
            if a<b:
                print(a, b)

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
            l = np.array(self.major_lengths)/len(self.major_lengths)
            n, bins, patches = ax.hist(l, num_bins)
            plt.show()


    def getEccentricity(self, nm=80, threshold=8000, display=True):
        """

        :param nm: the number of bins
        :param threshold: the threshold of rock size in pixel
        :param display:
        :return:
        """
        self.ecc = []
        for id in self.ids:
            bb = self.instances[id]['bb']
            mask = self.instances[id]['mask']
            image = self.tif[bb[0]:bb[2], bb[1]:bb[3], :]
            top_left = bb[:2]
            mask = mask - top_left
            mask = self.__create_bool_mask(mask, image.shape[:2])
            #_, contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # one of these two works
            contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            areas = [cv2.contourArea(cnt) for cnt in contours]
            size = np.max(areas)

            if size > threshold:
                continue

            i = np.argmax(areas)
            cnt = contours[i]
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse

            a = ma / 2
            b = MA / 2
            eccentricity = sqrt(pow(a, 2) - pow(b, 2))
            eccentricity = round(eccentricity / a, 2)

            self.ecc.append(eccentricity)


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
            l = np.array(self.ecc)
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

    def getOrientationHist(self, nm=20, threshold=0.6, display="polar"):
        self.orientations = []
        for id in self.ids:
            bb = self.instances[id]['bb']
            mask = self.instances[id]['mask']
            image = self.tif[bb[0]:bb[2], bb[1]:bb[3], :]
            top_left = bb[:2]
            mask = mask - top_left
            mask = self.__create_bool_mask(mask, image.shape[:2])
            # _, contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # one of these works
            contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            areas = [cv2.contourArea(cnt) for cnt in contours]
            i = np.argmax(areas)
            cnt = contours[i]
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            a = ma / 2
            b = MA / 2
            eccentricity = sqrt(pow(a, 2) - pow(b, 2))
            eccentricity = round(eccentricity / a, 2)
            self.orientations.append([angle, eccentricity])

            """
            #image = cv2.drawContours(image, cnt, -1, (0, 100, 0), 1)
            image = cv2.ellipse(image, ellipse, (0, 100, 0), 1)
            d = angle*pi/180
            start = (int(x - a*sin(d)), int(y + a*cos(d)))
            end = (int(x + a*sin(d)), int(y - a*cos(d)))
            print(start)
            print(end)
            image = cv2.line(image, start, end, (200, 200, 0), 1)
            print(eccentricity)
            print(x,y)
            print(MA,ma)
            print(angle)
            plt.imshow(image)
            plt.show()
            """

        if display == "polar":
            """
            the polar is not accurate because pi is not accurate
            """
            rad90 = 90 / 180.0 * np.piregisterArea
            nm = nm*4
            orn = np.asarray(self.orientations).copy()
            angles = orn[:,0]/180.0*np.pi

            for i,angle in enumerate(angles):
                if angle>=rad90:
                    angles[i] = angle - rad90
                else:
                    angles[i] = rad90 - angle

            ax = plt.subplot(111, projection='polar')
            bins = np.linspace(0.0, 2*np.pi, nm)
            n, bins_, patches = ax.hist(angles, bins)
            plt.show()

        elif display == "cart":
            fig, ax = plt.subplots()
            orn = np.asarray(self.orientations).copy()
            angles = orn[:, 0]
            for i, angle in enumerate(angles):
                if angle>=90:
                    angles[i] = angle - 90
                else:
                    angles[i] = 90 - angle
            n, bins, patches = ax.hist(angles, nm)
            plt.show()

        elif display == "polar2":
            fig, ax = plt.subplots()
            orn = np.asarray(self.orientations).copy()
            angles = orn[:, 0]

            for i, angle in enumerate(angles):
                if angle >= 90:
                    angles[i] = 270 - angle
                else:
                    angles[i] = 90 - angle

            n, bins, patches = ax.hist(angles, nm)

            print(n)
            print(bins)


            plt.close('all')

            fig, ax = plt.subplots()
            lim = np.max(n) + 20
            plt.xlim(-lim,lim)
            plt.ylim(0,lim)
            patches = []
            y0 = x0 = 0

            #image = np.zeros((y0,y0,3), dtype='uint8')
            #c = np.array([np.cos(np.deg2rad(x)) for x in range(0,95,10)])
            #s = np.array([np.sin(np.deg2rad(x)) for x in range(0,95,10)])
            colors = []

            for i in range(nm):
                angle1 = bins[i+1]
                angle2 = bins[i]
                r = n[i]
                x1 = r*np.cos(np.deg2rad(angle1))
                y1 = r*np.sin(np.deg2rad(angle1))
                x2 = r*np.cos(np.deg2rad(angle2))
                y2 = r*np.sin(np.deg2rad(angle2))
                pts = np.array(((x0,y0),(x1,y1),(x2,y2)))
                poly = plt.Polygon(pts, color='blue')
                patches.append(poly)
                colors.append(1 - r/lim)

            p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
            p.set_array(np.array(colors))
            ax.add_collection(p)

            plt.show()






    def __getPolygonPoints(self):
        pass


if __name__  ==  "__main__":
    ca = contourAnalysis()
    ca.readTiff("../C3.tif")
    ca.readInstances("../refined_instances.pickle")
    #ca.refineInstances("../shifted_contour.jpg")

    ca.registerArea(4146294, 4145785)  # 1. entire area
    #ca.registerArea(4146177, 4146113, 372380, 372490)  # selected area
    #ca.saveRegisteredInstances('talk.pickle')
    #ca.getSizeHist(threshold=4000)  # 2. get size hist
    #ca.registerArea(4146294, 4146244)
    #ca.getSizeHist()
    #ca.getOrientationHist(nm=15, display='polar2')  # 3. get orientation hist
    #ca.getMajorLengthHist(threshold=4000)

    ca.getEccentricity(nm=20, threshold=4000)