#!/usr/bin/env python
"""
strike_hist.py (python3)
Zhiang Chen, June 2019

Copyright (c) 2019 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU
"""

import os
import cv2
import gdal
import pickle
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from math import sqrt

class Strike_analysis(object):
    def __init__(self, skeleton_path, orth_path, instance_path, ):
        assert os.path.isfile(skeleton_path)
        assert os.path.isfile(orth_path)
        assert os.path.isfile(instance_path)

        self.skeleton = cv2.imread(skeleton_path)[:, :, 0]
        sk = np.max(self.skeleton, axis=1)
        sk = np.where(sk > 10)[0]
        self.sk_up, self.sk_down = sk[0], sk[-1]

        self.orth = gdal.Open(orth_path)
        coord = self.orth.GetGeoTransform()
        self.x, self.x_size, _, self.y, _, self.y_size = coord
        self.h, self.w = self.orth.GetRasterBand(1).ReadAsArray().shape
        self.X = self.x + self.x_size * self.w
        self.Y = self.y + self.y_size * self.h

        with open(instance_path, 'rb') as pk:
            self.instances = pickle.load(pk)

    def create_heatmap(self):
        B = self.orth.GetRasterBand(1).ReadAsArray()
        G = self.orth.GetRasterBand(2).ReadAsArray()
        R = self.orth.GetRasterBand(3).ReadAsArray()
        self.histmap = np.zeros((self.h, self.w, 3))
        self.histmap[:, :, 0], self.histmap[:, :, 1], self.histmap[:, :, 2] = R, G, B  # B, G, R

    def get_normal_vector(self, latitude_up, latitude_down):
        """
        get normal vector of the skeleton given latitude bound
        :param latitude_up:
        :param latitude_down:
        :return: intersection, norm vector (both in pixel coordinates)
        """
        pixel_up = self.__lad2pixel(latitude_up)
        pixel_down = self.__lad2pixel(latitude_down)
        assert self.sk_up < pixel_up < pixel_down < self.sk_down
        latitude = (latitude_up + latitude_down)/2
        intersection = self.__get_intersection(latitude)

        section = self.skeleton[pixel_up:pixel_down, :]
        points = np.where(section > 10)
        points = np.array(points)
        points = np.swapaxes(points, axis1=0, axis2=1)
        points[:, 0] = points[:, 0] + pixel_up
        a, b = np.polyfit(points[:, 1], points[:, 0], deg=1)  # y = a*x + b
        # Note the coordinates are the pixel coordinates
        norm_v = np.array((-1/a, 1.0))
        norm_v = norm_v / np.linalg.norm(norm_v)  # (y, x)

        return intersection, norm_v

    def generate_one_box(self, center, normal, offset, h, w):
        """
        generate one box that is 'offset' off the 'center' along 'normal' vector with height 'h' and width 'w'
        :param center: in pixel coordinates
        :param normal: in pixel coordinates
        :param offset: in pixel
        :param h: in pixel
        :param w: in pixel
        :return: box coordinates in pixel coordinates, top_left(y, x), top_right(y, x), bottom_right(y, x), bottom_left(y, x)
        """
        p1 = (-h / 2.0, -w / 2.0)
        p2 = (-h / 2.0, w / 2.0)
        p3 = (h / 2.0, w / 2.0)
        p4 = (h / 2.0, -w / 2.0)
        points = np.array((p1, p2, p3, p4))
        s, c = normal
        s, c = -c, s
        r = np.array(((c, -s), (s, c)))
        points_ = np.matmul(points, r)

        """
        print(normal)
        print(points)
        print(points_)
        plt.plot(points[:, 1], points[:, 0], 'x')
        plt.plot(points_[:, 1], points_[:, 0], 'ro')
        plt.plot((0, normal[1]*10), (0, normal[0]*10), 'o')
        plt.ylim(-100, 100)
        plt.xlim(-100, 100)
        plt.show()
        """
        points_ = points_ + center + offset*normal
        return points_.astype(int)

    def generate_boxes(self, center, normal, h, w, step, num, symmetric=True):
        """
        generate boxes
        :param center: pixel coordinates
        :param normal: pixel coordinates
        :param h: in pixel
        :param w: in pixel
        :param step: in pixel
        :param num: box number on one side
        :param symmetric: bool
        :return: boxes in pixel coordinates
        """
        offsets = [step*i for i in range(num+1)[1:]]

        if symmetric:
            offsets = [step * i for i in range(-num, num + 1)]

        boxes = [self.generate_one_box(center, normal, offset, h, w) for offset in offsets]
        return boxes

    def get_instances_in_box(self, box):
        """
        get instances in a box defined by points
        :param box:
        :return: instances in the box, list
        """
        polygon = Polygon(box)
        boxed_instances = []
        for instance in self.instances:
            y, x, Y, X = instance['bb']
            point = Point((y+Y)/2.0, (x+X)/2.0,)
            if polygon.contains(point):
                boxed_instances.append(instance)
        #print(len(boxed_instances))
        return boxed_instances

    def draw_box_in_skeleton(self, points, sk, color=255):
        """
        example:
            intersection, norm_v = sa.get_normal_vector(4146290, 4146285)
            points = sa.generate_one_box(intersection, norm_v, 0, 30, 100)
            box_sk = sa.draw_box_in_skeleton(points, sa.skeleton)
            cv2.imwrite("box_sk.jpg", box_sk)
        :param points:
        :param sk:
        :return:
        """
        n, _ = points.shape
        points = points.reshape((1, n, 2))
        box = points.copy()
        box[:, :, 0], box[:, :, 1] = points[:, :, 1], points[:, :, 0]
        sk_box = cv2.fillPoly(sk, box, color)
        return sk_box

    def draw_instances_in_skeleton(self, instances, sk):
        for instance in instances:
            mask = instance['mask']
            for y,x in mask:
                sk[y, x] = 255
        return sk

    def __get_intersection(self, latitude):
        """
        get an intersection with the skeleton given a latitude
        :param latitude: latitude in meters
        :return: intersection
        """
        pixel_y = self.__lad2pixel(latitude)
        assert (pixel_y >= 0) & (pixel_y < self.h)
        x = self.skeleton[pixel_y, :]
        pixel_x = np.where(x > 10)
        assert len(pixel_x[0].tolist()) > 0
        pixel_x = int(np.average(pixel_x))
        intersection = np.array((pixel_y, pixel_x)).astype(int)
        return intersection

    def __utm2pixel(self, utm_y, utm_x):
        """
        convert utm to pixel coordinates
        :param utm_x:
        :param utm_y:
        :return:
        """
        y = (utm_y - self.y) / self.y_size
        x = (utm_x - self.x) / self.x_size
        pixel_coord = np.array((y, x)).astype(int)
        return pixel_coord

    def __lad2pixel(self, utm_y):
        """
        convert utm latitude to pixel coordinate
        :param utm_y:
        :return:
        """
        y = (utm_y - self.y) / self.y_size
        return int(y)

    def __pixel2lad(self, pixel):
        utm_y = self.y + pixel * self.y_size
        return utm_y

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

    def __add_hist2colormap(self, hist, box, side_box, box_id = None, box_num=None, width_vector=None):
        scale = 1 # the scale of side box
        num = len(hist)
        p1 = box[0]
        p4 = box[3]
        direction = box[1] - box[0]
        direction_step = direction / num
        if not side_box:
            for i in range(num):
                b1 = p1 + direction_step * i
                b2 = b1 + direction_step
                b4 = p4 + direction_step * i
                b3 = b4 + direction_step
                subbox = np.array((b1, b2, b3, b4), dtype=int)
                subbox = subbox.reshape((1, 4, 2))
                points = subbox.copy()
                subbox[:, :, 0], subbox[:, :, 1] = points[:, :, 1], points[:, :, 0]
                self.histmap = cv2.fillPoly(self.histmap, subbox, (int(hist[i]), 0, int(hist[i])))
                #self.histmap = cv2.fillPoly(self.histmap, subbox, int(i*255/num) )
        elif side_box:
            direction_step = direction_step * scale

            p1 = p1 + width_vector * (box_num - box_id) * scale + width_vector * box_id            + width_vector*0.1
            p4 = p4 + width_vector * (box_id + 1) + width_vector * (box_num - box_id - 1) * scale  + width_vector*0.1

            for i in range(num):
                b1 = p1 + direction_step * i
                b2 = b1 + direction_step
                b4 = p4 + direction_step * i
                b3 = b4 + direction_step
                subbox = np.array((b1, b2, b3, b4), dtype=int)
                subbox = subbox.reshape((1, 4, 2))
                points = subbox.copy()
                subbox[:, :, 0], subbox[:, :, 1] = points[:, :, 1], points[:, :, 0]
                self.histmap = cv2.fillPoly(self.histmap, subbox, (int(hist[i]), 0, int(hist[i])))



    def __apply_mask(self, image, mask, alpha=0.5):
        """
        Apply the given mask to the image.
        :param image:
        :param mask:
        :param color: (0,0,1): blue
        :param alpha:
        :return:
        """
        for c in range(3):
            image[:, :, c] = image[:, :, c] * (1 - alpha) + mask[:, :, c] * alpha
        return image

    def rock_hist(self, boxes, size_max=4000, size_step=200, mode="size", side_box=False):
        """
        get the histograms of rocks that are in the boxes
        :param boxes:
        :return:
        """
        """ analysis"""
        rock_statistics = []
        for box in boxes:
            rock_box = dict()
            rock_box["box"] = box
            rocks = self.get_instances_in_box(box)
            rock_sizes = []
            orientations = []
            major_lengths = []
            for rock in rocks:
                bb = rock["bb"]
                mask = rock["mask"]
                top_left = bb[:2]
                mask = mask - top_left
                mask_shape = (bb[2] -bb[0], bb[3] - bb[1])
                mask = self.__create_bool_mask(mask, mask_shape)
                #_, contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # cv2.findContours has been changed in Ubuntu 18
                contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                areas = [cv2.contourArea(cnt) for cnt in contours]
                rock_size = np.max(areas)
                rock_sizes.append(rock_size)

                i = np.argmax(areas)
                cnt = contours[i]
                ellipse = cv2.fitEllipse(cnt)
                (x, y), (MA, ma), angle = ellipse
                a = ma / 2
                b = MA / 2
                eccentricity = sqrt(pow(a, 2) - pow(b, 2))
                eccentricity = round(eccentricity / a, 2)
                orientations.append([angle, eccentricity])
                major_lengths.append(2*a)
            rock_box["sizes"] = rock_sizes
            rock_box["orientations"] = orientations
            rock_box["major_lengths"] = major_lengths
            rock_statistics.append(rock_box)

        """ histogram"""
        size_bins = np.arange(0, size_max, size_step)
        length_bins = np.arange(0, 200, 10)
        box_nm = len(rock_statistics)
        box = rock_statistics[0]["box"]
        width_v = box[0]-box[3]
        for i, rock_box in enumerate(rock_statistics):
            if mode == "size":
                sizes = rock_box["sizes"]
                hist, _ = np.histogram(sizes, size_bins)
                hist = (np.array(hist) / np.max(hist) * 255).tolist()  # normalize histogram
                box = rock_box["box"]
                self.__add_hist2colormap(hist, box, side_box, i, box_nm, width_v)
            elif mode == "major_length":
                major_lengths = rock_box["major_lengths"]
                hist, _ = np.histogram(major_lengths, length_bins)
                hist = (np.array(hist) / np.max(hist) * 255).tolist()  # normalize histogram
                box = rock_box["box"]
                self.__add_hist2colormap(hist, box, side_box, i, box_nm, width_v)
            else:
                print("Undefined mode!")



    def overlap_hist_orth(self, histmap):
        B = self.orth.GetRasterBand(1).ReadAsArray()
        G = self.orth.GetRasterBand(2).ReadAsArray()
        R = self.orth.GetRasterBand(3).ReadAsArray()
        orth = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        orth[:, :, 0], orth[:, :, 1], orth[:, :, 2] = R, G, B
        del R, G, B
        overlap = self.__apply_mask(orth, histmap)
        return overlap


def hist_entire_scarp(sa):
    step = 30
    begin = 4146290
    sa.create_heatmap()

    for i in range(16):
        #intersection, norm_v = sa.get_normal_vector(begin - i*step, begin - i*step - step)
        intersection, norm_v = sa.get_normal_vector(begin - (15-i) * step, begin - (15-i) * step - step)

        boxes = sa.generate_boxes(intersection, norm_v, h=200, w=1600, step=200, num=4)
        sa.rock_hist(boxes)
        sa.rock_hist(boxes, mode="major_length", side_box=True)
        histmap = sa.histmap.astype(np.uint8).copy()
        cv2.imwrite("../contours/histmap.jpg", histmap)
        print(i)

    return sa.histmap.astype(np.uint8).copy()

if __name__ == "__main__":
    sa = Strike_analysis("../contours/resized_pruned_sk.jpg", "../C3.tif", "../registered_instances_v3.pickle")
    histmap = hist_entire_scarp(sa)
    #intersection, norm_v = sa.get_normal_vector(4146290, 4146260)
    """
    sk = sa.skeleton.copy()
    points = sa.generate_one_box(intersection, norm_v, 100, 30, 100)
    sk = sa.draw_box_in_skeleton(points, sk)
    cv2.imwrite("../box_sk.jpg", sk)
    """

    boxes = sa.generate_boxes(intersection, norm_v, h=200, w=1600, step=200, num=4)
    """
    sk = sa.skeleton.copy()
    for box in boxes:
        rocks = sa.get_instances_in_box(box)
        sk = sa.draw_box_in_skeleton(box, sk)
        #sk = sa.draw_instances_in_skeleton(rocks, sk)
    #cv2.imwrite("../contours/box_sk.jpg", sk)
    """
    #sa.create_heatmap()
    #sa.rock_hist(boxes)
    #sa.rock_hist(boxes, mode="major_length", side_box=True)
    #histmap = sa.histmap.astype(np.uint8).copy()
    cv2.imwrite("../contours/histmap.jpg", histmap)

    #del sa
    #sa = Strike_analysis("../contours/resized_pruned_sk.jpg", "../C3.tif", "../registered_instances_v3.pickle")
    #histmap = cv2.imread("../contours/histmap.jpg")
    #overlap = sa.overlap_hist_orth(histmap)
    #cv2.imwrite("../contours/overlap_hist_orth.jpg", overlap)
