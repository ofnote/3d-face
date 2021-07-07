"""
    Resource :- https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import numpy as np
import pandas as pd
import os
import glob
from PIL import Image
import scipy.io
from tqdm import tqdm
# from ..utils import util
import csv
import cv2

dataPath = os.getcwd()+'/3d-face/decalib/datasets'+"/300W_LP"
dirs = ['IBUG','AFW', 'AFW_Flip', 'HELEN', 'HELEN_Flip', 'IBUG_Flip', 'LFPW', 'LFPW_Flip']
# dirs = ['IBUG']

files = []
# fields = ['pt2d', 'roi', 'Illum_Para', 'Color_Para',
#           'Tex_Para', 'Shape_Para', 'Exp_Para', 'Pose_Para']
headers = ['image_name']
f = open('data.csv','a+')
writer = csv.writer(f)
writer.writerow(headers)

i = 0
for dir in dirs:
    print(dataPath+'/'+dir)
    for file in tqdm(os.listdir(dataPath+'/'+dir)):
        # print(len(file))
        # if(i >= 1):
        #     temp = pd.read_csv('./data.csv')
        #     if(len(temp) == 15):
        #         break
        if(file.endswith('.jpg')):
            arr = []
            arr.append(dir+'/'+file)
            writer.writerow(arr)

f.close()
