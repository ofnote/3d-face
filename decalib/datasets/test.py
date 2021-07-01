import scipy.io
import scipy
from PIL import Image
import pprint 
import numpy as np
import cv2
mat = scipy.io.loadmat('./300W_LP/AFW/AFW_134212_1_0.mat')
img = Image.open('./300W_LP/AFW/AFW_134212_1_0.jpg')
img = np.array(img)
fields = ['pt2d', 'roi', 'Illum_Para', 'Color_Para', 'Tex_Para', 'Shape_Para', 'Exp_Para', 'Pose_Para']
print(img.shape)
print(mat['pt2d'].shape)






# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# for i in range(68):
#     # print(landmarks2d[0][i],landmarks2d[1][i])
#     cv2.circle(img,(int(landmarks2d[0][i]),int(landmarks2d[1][i])),2,(255,0,0),2)

# cv2.imshow('img',img)
# cv2.waitKey(0)