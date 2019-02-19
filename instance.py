"""
instance.py (python3)
Zhiang Chen, Feb 2019

Copyright (c) 2019 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU

merge and refine all rocks, and represent them in global pixel coordinates
"""
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import time

class rock(object):
    def __init__(self):
        self.instances = []
        self.registered_instances = []

    def loadFile(self, f, mode="pickle"):
        """
        load file
        :param f: file
        :param mode:
        :return: instances
        """
        if mode == "pickle":
            with open(f, 'rb') as pk:
                instances = pickle.load(pk)
                return instances

    def saveRegisteredInstances(self, f='registered_instances.pickle', mode="pickle"):
        if mode == "pickle":
            with open(f, 'wb') as pk:
                pickle.dump(self.registered_instances, pk)

    def addAsGlobalInstances(self, instances, mask_resize=None, swap_coord=True):
        """
        get instances in global pixel coordinates, and store as self.instances
        :param instances: instances, list
        :param mask_resize: the size of mask to be resized
        :param swap_coord: if the coordinates (upper left coordinates which images the instances belong to)
        of the instances need to be swapped
        :return: None
        """
        for instance in instances:
            mask = instance['mask']
            dim = mask.shape
            bb = instance['bb']
            coord = instance['coord']
            if mask_resize != None:
                mask = cv2.resize(mask.astype(float), mask_resize).astype(bool)
                scale_x = float(mask_resize[0])/dim[0]
                scale_y = float(mask_resize[1])/dim[1]
                bb = [int(bb[0]*scale_x), int(bb[1]*scale_y), int(bb[2]*scale_x), int(bb[3]*scale_y)]

            if swap_coord:
                coord = [coord[1], coord[0]]

            coord = np.asarray(coord + coord).astype(int)
            bb = np.asarray(bb).astype(int) + coord
            mask_coord = np.argwhere(mask==True) + coord[:2]

            global_inst = dict()
            global_inst['bb'] = bb
            global_inst['mask'] = mask_coord
            global_inst['coord'] = coord[:2]

            self.instances.append(global_inst)

    def resetInstances(self):
        """
        reset self.instances
        :return:
        """
        self.instances = []

    def refineInstance(self, overlap_x, overlap_y, tile_size, clear_register=False):
        """
        refine and merge all instances, and then register all instances
        :param tile_size: tile size
        :param clear_register: True if the register needs to be cleared
        :return:
        """
        self.tile_scope = tile_size + 100
        self.tile_size = tile_size
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y


        if clear_register:
            self.registered_instances = []

        for i,instance in enumerate(self.instances):
            id = self.__checkRegister(instance)

            if id == None:
                self.registered_instances.append(instance)
            else:
                #print("merging")
                self.__mergeInstances(id, instance)
            #if i%100 == 0:
            #print(len(self.registered_instances))



    def __checkRegister(self, instance, threshold=5):
        """
        naive brute force search if instance can be merged to any one in self.registered_instances
        :param instance: instance to check
        :return: id of the one to be merge in self.registered_instances, otherwise None
        """
        if self.__checkBoundary(instance): # True when the instance is not on the boundary
            return None

        for id, ri in enumerate(self.registered_instances):
            if self.__checkCoords(ri['coord'], instance['coord']):
                if self.__checkBBOverlap(ri['bb'], instance['bb']):
                    nb = self.__checkMaskOverlap(ri['mask'], instance['mask'])
                    if nb > threshold:
                        return id

        return None

    def __checkBoundary(self, instance):
        """
        check if the instance is on the boundary
        :param instance:
        :return: True if it is not; False if it is
        """
        bb = instance['bb']
        coord = instance['coord']
        if (bb[0] - coord[0]) <= self.overlap_y:
            return False
        elif (bb[1] - coord[1]) <= self.overlap_x:
            return False
        elif (coord[0] + self.tile_size - bb[2]) <= self.overlap_y:
            return False
        elif (coord[1] + self.tile_size - bb[3]) <= self.overlap_x:
            return False
        else:
            return True



    def __checkBBOverlap(self, bb1, bb2):
        [xmin_a, ymin_a, xmax_a, ymax_a] = bb1.tolist()
        [xmin_b, ymin_b, xmax_b, ymax_b] = bb2.tolist()
        if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
            return True
        elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
            return True
        elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
            return True
        elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
            return True
        else:
            return False




    def __checkCoords(self, ri_coord, coord):
        """
        check if the instance can be added to the registered instance by coords
        :param ri_coord: registered_instance coordinates
        :param coord: instance coordinate
        :return: True if it can be merged
        """

        """
        Currently only consider a simplest situation that one rock cannot be merged to any rock from the same tile
        """
        diff = ri_coord.reshape(-1, 2) - coord.reshape(2)
        if (np.abs(np.min(diff)) < self.tile_scope) & (np.abs(np.max(diff)) < self.tile_scope):
            coord_ls1 = ri_coord.reshape(-1, 2).tolist()
            coord_ls2 = coord.reshape(2).tolist()
            if coord_ls2 in coord_ls1:
                return False
            return True
        else:
            return False


    def __checkMaskOverlap(self, mask1, mask2):
        """
        check the overlap between two masks
        :param mask1:
        :param mask2:
        :return: overlap pixel number
        """
        mask_1 = mask1.tolist()
        mask_2 = mask2.tolist()
        overlap = [px for px in mask_1 if px in mask_2]
        return(len(overlap))


    def __mergeInstances(self, id, instance):
        ri = self.registered_instances[id]
        ri['mask'] = np.concatenate((ri['mask'], instance['mask']), axis=0)
        ri['coord'] = np.concatenate((ri['coord'].reshape(-1,2), instance['coord'].reshape(-1,2)), axis=0)
        min_y = np.min(ri['mask'][:, 0])
        min_x = np.min(ri['mask'][:, 1])
        max_y = np.max(ri['mask'][:, 0])
        max_x = np.max(ri['mask'][:, 1])
        ri['bb'] = np.array((min_y, min_x, max_y, max_x))
        self.registered_instances[id] = ri



if __name__  ==  "__main__":
    rk = rock()

    t1 = time.time()
    instances = rk.loadFile("./datasets/C3/pickles/instances_0.pickle")
    rk.addAsGlobalInstances(instances, (400, 400))
    rk.refineInstance(10, 10, 400)
    t2 = time.time()
    print(t2-t1)


    rk.resetInstances()
    instances = rk.loadFile("./datasets/C3/pickles/instances_1.pickle")
    rk.addAsGlobalInstances(instances, (400, 400))
    rk.refineInstance(10, 10, 400)
    t2 = time.time()
    print(t2 - t1)

    rk.resetInstances()
    instances = rk.loadFile("./datasets/C3/pickles/instances_2.pickle")
    rk.addAsGlobalInstances(instances, (400, 400))
    rk.refineInstance(10, 10, 400)
    t2 = time.time()
    print(t2 - t1)

    rk.resetInstances()
    instances = rk.loadFile("./datasets/C3/pickles/instances_3.pickle")
    rk.addAsGlobalInstances(instances, (400, 400))
    rk.refineInstance(10, 10, 400)
    t2 = time.time()
    print(t2 - t1)

    rk.resetInstances()
    instances = rk.loadFile("./datasets/C3/pickles/instances_4.pickle")
    rk.addAsGlobalInstances(instances, (400, 400))
    rk.refineInstance(10, 10, 400)
    t2 = time.time()
    print(t2 - t1)

    rk.resetInstances()
    instances = rk.loadFile("./datasets/C3/pickles/instances_5.pickle")
    rk.addAsGlobalInstances(instances, (400, 400))
    rk.refineInstance(10, 10, 400)
    t2 = time.time()
    print(t2 - t1)

    rk.resetInstances()
    instances = rk.loadFile("./datasets/C3/pickles/instances_6.pickle")
    rk.addAsGlobalInstances(instances, (400, 400))
    rk.refineInstance(10, 10, 400)
    t2 = time.time()
    print(t2 - t1)

    rk.resetInstances()
    instances = rk.loadFile("./datasets/C3/pickles/instances_7.pickle")
    rk.addAsGlobalInstances(instances, (400, 400))
    rk.refineInstance(10, 10, 400)
    t2 = time.time()
    print(t2 - t1)
    print(len(rk.registered_instances))

    rk.saveRegisteredInstances()

    #"""
    #a = np.array(((10,10),(20,30)))
    #b = np.array(((15,15),(30,20)))
    #c = rk.checkBB(b,a)
    #print(c)
    #"""



"""
n = range(10000,80000,10000)
t = [105.3374490737915,
354.87612986564636,
744.066376209259,
1261.9850449562073,
1929.37216091156,
2737.7477049827576,
3638.1695008277893]
plt.plot(n,t)
plt.show()
"""

