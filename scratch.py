# from numba import cuda
import numpy as np
import glob
import cv2
from time import time
import matplotlib.pyplot as plt
from multiprocessing import Pool, Process


paths = glob.glob('./work/train_frames/*')





num_imgs = 1664

def read_paths(path, cx, cy):
    img = cv2.imread(path)[:cx, :cy]
    return img, paths.index(path)


if __name__ =='__main__':


    start = time()
    args = []

    for i in range(num_imgs):
        args.append((paths[i], 128, 128))
    with Pool(8) as pool:  # 4 is number of processes we want to use
        results = list(pool.starmap(read_paths, args))
        # print(results)
    print(time()- start)


    start = time()
    args = []

    for i in range(num_imgs):
        args.append((paths[i], 128, 128))
    with Pool(4) as pool:  # 4 is number of processes we want to use
        results = list(pool.starmap(read_paths, args))
        # print(results)
    print(time()- start)
    # start = time()
    # result2 = [read_paths(i, 128, 128) for i in paths[:num_imgs]]
    # print(time() - start)