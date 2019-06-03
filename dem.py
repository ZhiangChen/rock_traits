#!/usr/bin/env python
"""
dem.py (python2)
Zhiang Chen, May 2019

Copyright (c) 2019 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU

"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import elevation
import richdem as rd
import cv2
from skimage import morphology
from skimage import transform
from skimage.util import invert
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
import scipy.ndimage
import gdal


class DEM(object):
    def __init__(self, file_path):
        assert os.path.isfile(file_path)

        self.dem_data = rd.LoadGDAL(file_path)
        self.shape = self.dem_data.shape

    def smooth(self):

        self.dem_data = cv2.blur(self.dem_data, (50, 50))
        self.dem_data = cv2.pyrDown(self.dem_data)
        self.dem_data = cv2.blur(self.dem_data, (35, 35))
        self.dem_data = cv2.pyrDown(self.dem_data)
        self.dem_data = cv2.blur(self.dem_data, (25, 25))
        self.dem_data = cv2.pyrDown(self.dem_data)
        self.dem_data = cv2.blur(self.dem_data, (15, 15))
        #self.dem_data = cv2.GaussianBlur(self.dem_data, (5, 5), 0)
        #self.dem_data = cv2.blur(self.dem_data, (5, 5))
        self.dem_data = cv2.pyrUp(self.dem_data)
        self.dem_data = cv2.pyrUp(self.dem_data)
        self.dem_data = cv2.pyrUp(self.dem_data)
        self.dem_data = transform.resize(self.dem_data, self.shape)



    def display_contour(self, levels=list(range(0, 50, 5)), fill_color=True):
        #dem = np.fliplr(self.dem_data)
        dem = np.flipud(self.dem_data)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if fill_color:
            ct = plt.contourf(dem, cmap="viridis", levels=levels)
        else:
            ct = plt.contour(dem, cmap="viridis", levels=levels)
        plt.title("Elevation Contours of Bishop C3")
        cbar = plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        return ct

    def transform_slope(self):
        self.dem_data = rd.rdarray(self.dem_data, no_data=-9999)
        self.dem_data = rd.TerrainAttribute(self.dem_data, attrib='slope_riserun')


    def display_slope(self):
        self.dem_data = rd.rdarray(self.dem_data, no_data=-9999)
        rd.rdShow(self.dem_data, axes=False, cmap='magma', figsize=(8, 5.5))


    def display(self, dem):
        plt.imshow(dem)
        plt.colorbar()
        plt.show()

    def save_contour(self, ct, flip=True):  # matplotlib realization
        img = np.zeros(self.dem_data.shape, np.uint8)
        gray = 0
        gray_step = 255/len(ct.allsegs)
        for i, contours in enumerate(ct.allsegs):  # contours at the same elevation
            gray += gray_step
            print(gray)
            for j, contour in enumerate(contours):
                contour = contour.reshape((-1, 1, 2))
                img = cv2.polylines(img, np.int32(contour), True, gray)
        if flip:
            img = np.flipud(img)
        cv2.imwrite("contour.jpg", img)

    def save_contour2(self, ct, flip=True):  # matplotlib realization 2
        contours = []
        # for each contour line
        i = j = 0
        gray = 0
        gray_step = 255 / len(ct.allsegs)
        for cc in ct.collections:
            i += 1
            j = 0
            gray += gray_step
            paths = []
            # for each separate section of the contour line
            for pp in cc.get_paths():
                j += 1
                xy = []
                # for each segment of that section
                for vv in pp.iter_segments():
                    xy.append(vv[0])
                paths.append(np.vstack(xy))
                img = np.zeros(self.dem_data.shape, np.uint8)
                path = np.int32(np.vstack(xy))
                path = path.reshape((-1, 1, 2))
                img = cv2.polylines(img, path, True, gray)
                if flip:
                    img = np.flipud(img)
                cv2.imwrite("../contours/"+str(i)+'_'+str(j)+".jpg", img)

            contours.append(paths)

    def generate_contour(self, levels, save_single=True, save_all=True):
        dem = self.dem_data.copy()
        m = dem.max()
        n = dem.min()
        self.contours = []


        valid_levels = []
        for level in levels:
            if level < n:
                continue
            if level > m:
                break
            valid_levels.append(level)

        gray = 0
        gray_step = 255 / len(valid_levels)
        img = np.zeros(self.dem_data.shape, np.uint8)

        for level in valid_levels:
            gray += gray_step
            contour = (dem > level) * gray
            self.contours.append(contour)
            if save_single:
                cv2.imwrite("../contours/" + str(level) + ".jpg", contour)
            if save_all:
                img = np.maximum(img, contour)

        if save_all:
            cv2.imwrite("../contours/all_contours.jpg", img)

    def read_contour(self, file_path, threshold=10):
        assert os.path.isfile(file_path)
        return cv2.imread(file_path)[:,:,0] > threshold

    def read_orth(self, f):
        assert os.path.isfile(f)
        self.ds = gdal.Open(f)
        R = self.ds.GetRasterBand(1).ReadAsArray()
        G = self.ds.GetRasterBand(2).ReadAsArray()
        B = self.ds.GetRasterBand(3).ReadAsArray()
        dim = R.shape
        resize = (dim[0], dim[1], 1)
        self.orth = np.concatenate((R.reshape(resize), G.reshape(resize), B.reshape(resize)), axis=2)
        return self.orth.copy()

    def generate_skeleton(self, file_path):
        contour = self.read_contour(file_path)
        sk = morphology.medial_axis(contour)
        inv_sk = invert(sk)
        return inv_sk

    def refine_contour(self, contour, itr=2):
        _, paths, _ = cv2.findContours(contour.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(cnt) for cnt in paths]
        id = areas.index(max(areas))
        contour = paths[id]
        img = np.zeros(self.dem_data.shape, np.uint8)
        n, _, _ = contour.shape
        contour = contour.reshape((1, n, 2))
        contour = cv2.fillPoly(img, contour, 255)
        return contour

    def erosion_contour(self, contour, degree=30):
        selem = disk(degree)
        return erosion(contour, selem)

    def opening_contour(self, contour, degree=30):
        selem = disk(degree)
        return opening(contour, selem)

    def dilation_contour(self, contour, degree=30):
        selem = disk(degree)
        return dilation(contour, selem)

    def widen_contour(self, contour, delta=100):  # left
        h,w = contour.shape
        for i in range(h):
            left_pt = contour[i,:].argmax()
            if left_pt>delta:
                contour[i, left_pt-delta:left_pt] = 255

        return contour

    def prolong_contour_tail(self, contour, delta=100, tail=90):
        h, w = contour.shape
        t = 0
        for i in range(h)[1:]:
            colume = contour[-i, :] != 0
            if colume.sum() > 0:
                if i<delta:
                    delta_down = i
                else:
                    delta_down = delta
                contour[-i:-i+delta_down, :] = contour[-i, :]
                t += 1
                if t == tail:
                    break
        return contour

    def shiftlr_contour(self, contour, delta):  # shift image to left or right
        mh, nh, mw, nw = self.__bbox(contour)
        img = np.zeros(self.dem_data.shape, np.uint8)
        img[:, mw+delta:nw+delta] = contour[:, mw:nw]
        return img

    def prune_skeleton(self, skeleton):  # only keep the most left skeleton
        h, w = skeleton.shape
        img = np.zeros((h,w), dtype=np.uint8)
        for i in range(h):
            column = skeleton[i, :]
            p = np.where(column < 0.5)[0]
            if len(p.tolist()) != 0:
                img[i,p[0]] = 255
        return img



    def overlap_two_contours(self, contour_path1, contour_path2):
        ct1 = self.read_contour(contour_path1) * 255
        ct2 = self.read_contour(contour_path2) * 255
        size = list(self.dem_data.shape)
        size.append(3)
        img = np.zeros(size, np.uint8)
        img[:,:, 0] = ct1
        img[:,:, 1] = ct2
        return img

    def overlap_contour_dem(self, contour_path, dem_path):
        ct = self.read_contour(contour_path) * 255
        dem_data = rd.LoadGDAL(dem_path)
        m = np.max(dem_data)
        n = np.min(dem_data)
        dem_data = (dem_data - n)/float(m)*255

        size = list(dem_data.shape)
        size.append(3)
        img = np.zeros(size, np.uint8)
        img[:, :, 0] = ct
        img[:, :, 1] = dem_data
        return img

    def overlap_contour_orth(self, contour_path, orth_path):
        ct = (self.read_contour(contour_path)*255).astype(np.uint8)
        orth = self.read_orth(orth_path)
        h,w,_ = orth.shape
        if orth.shape[:2] != ct.shape:
            ct = cv2.resize(ct, dsize=(w,h), interpolation=cv2.INTER_NEAREST)

        ct = ct>10
        overlap = self.__apply_mask(orth, ct)
        cv2.imwrite("../contours/overlap_orth_contour.jpg", overlap)


    def __apply_mask(self, image, mask, color=[1,0,0], alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image


    def __bbox(self, img):
        a = np.where(img != 0)
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        return bbox  # min_h, max_h, min_w, max_w



if __name__  ==  "__main__":
    dem = DEM("../C3_dem.tif")
    # get contours
    #dem.transform_slope()
    #dem.smooth()
    #levels = list(np.arange(0, 4, 0.2))[6:]
    #dem.generate_contour(levels)

    """
    ct = dem.read_contour("../contours/1.6.jpg")
    ct = dem.refine_contour(ct)
    ct = dem.widen_contour(ct * 255)
    ct = dem.opening_contour(ct)
    ct = dem.refine_contour(ct)
    cv2.imwrite("../contours/refined0_contour.jpg", ct)
    ct = dem.erosion_contour(ct)
    ct = dem.refine_contour(ct)
    cv2.imwrite("../contours/refined1_contour.jpg", ct)
    ct = dem.opening_contour(ct)
    ct = dem.refine_contour(ct)
    cv2.imwrite("../contours/refined2_contour.jpg", ct)
    ct = dem.erosion_contour(ct)
    ct = dem.refine_contour(ct)
    cv2.imwrite("../contours/refined3_contour.jpg", ct)
    ct = dem.widen_contour(ct, delta=20)
    ct = dem.opening_contour(ct)
    ct = dem.refine_contour(ct)
    cv2.imwrite("../contours/refined4_contour.jpg", ct)
    """

    """
    # 1.6.jpg
    ct = dem.read_contour("../contours/refined4_contour.jpg")
    ct = np.fliplr(ct)
    ct = dem.widen_contour(ct * 255, delta=50)
    ct = np.fliplr(ct)
    ct = dem.prolong_contour_tail(ct, delta=120)
    ct = dem.dilation_contour(ct, degree=10)
    cv2.imwrite("../contours/refined5_contour.jpg", ct)
    # overlapping
    overlap = dem.overlap_two_contours("../contours/refined5_contour.jpg", "../contours/1.6.jpg")
    cv2.imwrite("../contours/overlap_contour.jpg", overlap)
    overlap = dem.overlap_contour_dem("../contours/refined5_contour.jpg", "../C3_dem.tif")
    cv2.imwrite("../contours/overlap_contour_dem.jpg", overlap)
    dem.overlap_contour_orth("../contours/refined5_contour.jpg", "../C3_mask_v3.tif")
    """

    """
    # shift contour
    ct = dem.read_contour("../contours/refined5_contour.jpg")
    ct = dem.shiftlr_contour(ct*255, 40)
    cv2.imwrite("../contours/shifted_contour.jpg", ct)
    overlap = dem.overlap_contour_dem("../contours/shifted_contour.jpg", "../C3_dem.tif")
    cv2.imwrite("../contours/overlap_contour_dem.jpg", overlap)
    dem.overlap_contour_orth("../contours/shifted_contour.jpg", "../C3_mask_v3.tif")
    """


    """
    # 1.8.jpg
    ct = dem.read_contour("../contours/1.8.jpg")
    ct = dem.refine_contour(ct*255)
    cv2.imwrite("../contours/refined6_contour.jpg", ct)
    overlap = dem.overlap_contour_dem("../contours/refined6_contour.jpg", "../C3_dem.tif")
    cv2.imwrite("../contours/overlap_contour_dem.jpg", overlap)
    dem.overlap_contour_orth("../contours/refined6_contour.jpg", "../C3_mask_v3.tif")
    """


    # matplotlib display contour example
    # ct = dem.display_contour((0, 1.5, 2, 2.5, 3, 3.5, 4))
    # ct = dem.display_contour((0, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3. , 3.2, 3.4, 3.6, 3.8, 4.))
    # dem.save_contour2(ct)

    # skeleton example
    inv_sk = dem.generate_skeleton("../contours/shifted_contour.jpg")
    cv2.imwrite("../contours/inv_sk.jpg", inv_sk*255)
    pruned_sk = dem.prune_skeleton(inv_sk)
    pruned_sk = dem.prolong_contour_tail(pruned_sk, delta=100, tail=1)
    pruned_sk = invert(pruned_sk)
    cv2.imwrite("../contours/pruned_sk.jpg", pruned_sk)
    overlap = dem.overlap_two_contours("../contours/pruned_sk.jpg", "../contours/shifted_contour.jpg")
    cv2.imwrite("../contours/sk_contour.jpg", overlap)


    # prolong example
    # ct = dem.prolong_contour_tail(ct*255)
    # cv2.imwrite("../contours/prolong_contour.jpg", ct)

