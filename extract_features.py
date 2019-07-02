# imports
import numpy as np                     # numeric python lib

import matplotlib.image as mpimg       # reading images to numpy arrays

# reading an image file using matplotlib into a numpy array


def feat(path):
    img = mpimg.imread(path)
    
    
    from skimage.feature import corner_harris
    
    coords = corner_harris(img)
    
    feat = coords.flatten()
    
    return feat

import os
color_img_dir = "D:\\Sanjeev\\Hand Gesture Recognition\\my\\"
color_img_files = os.listdir(color_img_dir)
data = []
label= []  
for color_img_file in color_img_files:
    color = os.listdir(color_img_dir+color_img_file)
    
    for p in color:
        path = color_img_dir+color_img_file+"\\"+p
        
        feature = feat(path)
    
        data.append(feature)
        label.append(color_img_file)

np.save("new_train1_data.npy",data)
np.save("new_train1_label.npy",label)
