#!/usr/bin/env python
"""
dem.py (python3)
Zhiang Chen, June 2019

Copyright (c) 2019 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU
"""

import os
import cv2
import gdal
import pickle


class Strike_analysis(object):
    def __init__(self, skeleton_path, orth_path, instance_path):
        assert os.path.isfile(skeleton_path)
        assert os.path.isfile(orth_path)
        assert os.path.isfile(instance_path)

        self.skeleton = cv2.imread(skeleton_path)
        self.orth = gdal.Open(orth_path)
        with open(instance_path, 'rb') as pk:
            self.instances = pickle.load(pk)

    def get_normal_vector(self, latitude):
        """
        get a normal vector of the skeleton at this latitude
        :param latitude: latitude in meters
        :return: intersection, normal_vector
        """
        pass

    def get_instances_in_box(self, center, h, w, normal_vector):
        pass

    def generate_boxes(self, ):
        pass

    


if __name__ == "__main__":
    sa = Strike_analysis("../contours/pruned_sk.jpg", "../C3_mask_v3.tif", "../registered_instances_v3.pickle")
