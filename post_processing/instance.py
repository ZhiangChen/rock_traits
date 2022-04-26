"""
instance.py (python3)
Zhiang Chen, Feb 2019

Copyright (c) 2019 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU

merge and refine all rocks, and represent them in global pixel coordinates
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

class rock(object):
    def __init__(self):
        self.instances = []
        self.non_edge_instances = []
        self.edge_instances = []

        self.added_rocks = 0
        self.merged_rocks = 0
        self.all_rocks = 0

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

    def readFault(self, path):
        self.fault = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def saveRegisteredInstances(self, f='registered_instances.pickle', mode="pickle"):
        registered_instances = self.non_edge_instances + self.edge_instances
        print(len(registered_instances))
        if mode == "pickle":
            with open(f, 'wb') as pk:
                pickle.dump(registered_instances, pk)

    def addAsGlobalInstances(self, instances, mask_resize=None, swap_coord=True, refine=True):
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
            bb = instance['bb']
            coord = instance['coord']
            dim = mask.shape

            if mask_resize != None:
                mask = cv2.resize(mask.astype(float), mask_resize).astype(bool)
                scale_x = float(mask_resize[0])/dim[0]
                scale_y = float(mask_resize[1])/dim[1]
                bb = [int(bb[0]*scale_x), int(bb[1]*scale_y), int(bb[2]*scale_x), int(bb[3]*scale_y)]

            if swap_coord:
                coord = [coord[1], coord[0]]

            coord = np.asarray(coord + coord).astype(int)
            bb = np.asarray(bb).astype(int) + coord
            mask = mask > 0.9
            mask_coord = np.argwhere(mask==True) + coord[:2]
            if len(mask_coord) < 2:
                continue

            x, y = mask_coord[:, 0].min(), mask_coord[:, 1].min()
            x_, y_ = mask_coord[:, 0].max(), mask_coord[:, 1].max()
            bb = np.array((x, y, x_, y_))

            if (x == x_) | (y ==y_):
                continue

            global_inst = dict()
            global_inst['bb'] = bb
            global_inst['mask'] = mask_coord
            global_inst['coord'] = coord[:2]

            if refine:
                x = int((bb[0] + bb[2]) / 2)
                y = int((bb[1] + bb[3]) / 2)
                if self.fault[x, y] == 255:
                    self.instances.append(global_inst)
            else:
                self.instances.append(global_inst)

    def resetInstances(self):
        """/media/sarah/4dbf89b0-d16f-446a-93da-11629c6d3348/sarah/Zhiang_mask_rcnn/data_augmentor
        reset self.instances
        :return:
        """
        self.instances = []

    def refineInstance(self, overlap_x, overlap_y, tile_size):
        """
        refine and merge all instances, and then register all instances
        :param tile_size: tile size
        :return:
        """
        self.tile_scope = tile_size + 100
        self.tile_size = tile_size
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y

        for i, instance in enumerate(self.instances):
            id = self.__checkRegister(instance)

            if id == None:
                self.non_edge_instances.append(instance)
                self.added_rocks += 1

            else:
                self.__mergeInstances(id, instance)



    def __checkRegister(self, instance, threshold=20):
        """
        1. check if the instance is on the boundary. If it is not, then it can be directly added to self.added_instances
        2. If it is, then check if it can be merged to any instances in self.merged_instances.
        3. If it can, return the instance id in self.merged_instances. If it cannot, return -1
        :param instance: instance to check
        :return: id of the one to be merge in self.registered_instances, otherwise None
        """
        if self.__checkBoundary(instance):  # True when the instance is not on the boundary
            return None

        for id, ri in enumerate(self.edge_instances):
            if self.__checkBBOverlap(ri['bb'], instance['bb']):
                nb = self.__checkMaskOverlap(ri['mask'], instance['mask'])
                if nb > threshold:
                    return id

        return -1

    def __checkBoundary(self, instance):
        """
        check if the instance is on the boundary
        :param instance:
        :return: True if it is not; False if it is
        """
        bb = instance['bb']
        coord = instance['coord']
        if (bb[0] - coord[0]) <= self.overlap_x:
            return False
        elif (bb[1] - coord[1]) <= self.overlap_y:
            return False
        elif (coord[0] + self.tile_size - bb[2]) <= self.overlap_x:
            return False
        elif (coord[1] + self.tile_size - bb[3]) <= self.overlap_y:
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
        return (len(overlap))


    def __mergeInstances(self, id, instance):
        if id == -1:
            self.edge_instances.append(instance)
            self.added_rocks += 1
        else:
            ri = self.edge_instances[id]
            ri['mask'] = np.concatenate((ri['mask'], instance['mask']), axis=0)
            ri['coord'] = np.concatenate((ri['coord'].reshape(-1,2), instance['coord'].reshape(-1,2)), axis=0)
            min_y = np.min(ri['mask'][:, 0])
            min_x = np.min(ri['mask'][:, 1])
            max_y = np.max(ri['mask'][:, 0])
            max_x = np.max(ri['mask'][:, 1])
            ri['bb'] = np.array((min_y, min_x, max_y, max_x))
            self.edge_instances[id] = ri
            self.merged_rocks += 1



if __name__  ==  "__main__":
    """
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
    
    rk.resetInstances()
    instances = rk.loadFile("./datasets/C3/pickles/instances_8.pickle")
    rk.addAsGlobalInstances(instances, (400, 400))
    rk.refineInstance(10, 10, 400)
    t2 = time.time()
    print(t2 - t1)

    rk.saveRegisteredInstances()
    """

    #"""
    #a = np.array(((10,10),(20,30)))
    #b = np.array(((15,15),(30,20)))
    #c = rk.checkBB(b,a)
    #print(c)
    #"""
    import sys

    #sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

    rk = rock()
    rk.readFault('./datasets/C3/fault.png')  # fault scarp mask image
    for i in range(9):
        print(i)
        name = "./datasets/C3/rocks_c3_rgbd1_%02d.pickle" % i
        instances = rk.loadFile(name)
        print(len(instances))
        rk.addAsGlobalInstances(instances, swap_coord=False, refine=False)
        print(len(rk.instances))
        rk.refineInstance(10, 10, 400)
        rk.resetInstances()

    rk.saveRegisteredInstances('registered_instances_c3_rgbd1.pickle')



"""
import matplotlib.pyplot as plt
n1 = range(0,80000,10000)
n2 = range(0,80000,10000)
t1 = [0, 
105.3374490737915,
354.87612986564636,
744.066376209259,
1261.9850449562073,
1929.37216091156,
2737.7477049827576,
3638.1695008277893]
t2= [0, 
31.482572078704834, 
52.84216094017029, 
85.81673884391785, 
161.5292031764984, 
219.15165948867798, 
273.3696777820587, 
340.45087242126465]
plt.plot(n2,t2)
plt.show()4680
"""

