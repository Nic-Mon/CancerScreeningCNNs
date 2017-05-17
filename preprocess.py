import numpy as  np
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import math
from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
from glob import glob
import os

import numpy as  np
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import math
from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
from glob import glob
import os

def maxHist(hist):
    maxArea = (0, 0, 0)
    height = []
    position = []
    for i in range(len(hist)):
        if (len(height) == 0):
            if (hist[i] > 0):
                height.append(hist[i])
                position.append(i)
        else: 
            if (hist[i] > height[-1]):
                height.append(hist[i])
                position.append(i)
            elif (hist[i] < height[-1]):
                while (height[-1] > hist[i]):
                    maxHeight = height.pop()
                    area = maxHeight * (i-position[-1])
                    if (area > maxArea[0]):
                        maxArea = (area, position[-1], i)
                    last_position = position.pop()
                    if (len(height) == 0):
                        break
                position.append(last_position)
                if (len(height) == 0):
                    height.append(hist[i])
                elif(height[-1] < hist[i]):
                    height.append(hist[i])
                else:
                    position.pop()    
    while (len(height) > 0):
        maxHeight = height.pop()
        last_position = position.pop()
        area =  maxHeight * (len(hist) - last_position)
        if (area > maxArea[0]):
            maxArea = (area, len(hist), last_position)
    return maxArea
            

def maxRect(img):
    maxArea = (0, 0, 0)
    addMat = np.zeros(img.shape)
    for r in range(img.shape[0]):
        if r == 0:
            addMat[r] = img[r]
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
        else:
            
            addMat[r] = img[r] + addMat[r-1]
            addMat[r][img[r] == 0] *= 0
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
    return (int(maxArea[3]+1-maxArea[0]/abs(maxArea[1]-maxArea[2])), maxArea[2], maxArea[3], maxArea[1], maxArea[0])

def cropCircle(img):
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))

    img = cv2.resize(img, dsize=tile_size)
            
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]
            
    ff = np.zeros((gray.shape[0],gray.shape[1]), 'uint8') 
    cv2.drawContours(ff, main_contour, -1, 1, 15)
    ff_mask = np.zeros((gray.shape[0]+2,gray.shape[1]+2), 'uint8')
    cv2.floodFill(ff, ff_mask, (int(gray.shape[1]/2), int(gray.shape[0]/2)), 1)
    #cv2.circle(ff, (int(gray.shape[1]/2), int(gray.shape[0]/2)), 3, 3, -1)
    
    rect = maxRect(ff)
    img_crop = img[min(rect[0],rect[2]):max(rect[0],rect[2]), min(rect[1],rect[3]):max(rect[1],rect[3])]
    cv2.rectangle(ff,(min(rect[1],rect[3]),min(rect[0],rect[2])),(max(rect[1],rect[3]),max(rect[0],rect[2])),3,2)
    
    return img_crop

# make image folders

from glob import glob
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_image_data(image_id, image):
    img = cv2.imread(image)
    #assert img is not None, "Failed to read image : %s" % (image_id)
    if img is None:
        print('image {} is None'.format(image_id))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    
    
    
standard_size=(224,224)
def img_resize(image):
    """
    takes an rgb image and it into an image of size 300,300,3 which is then flattened to 90000,3"""
    img = cv2.resize(image, standard_size)
    return img
    
x_test = []
#y_test = []
print('Read testing set images')

for i in range(512):
    x_test.append('test/{}.jpg'.format(i))

cropped_images=[]
for k, path in enumerate(x_test):
    img = get_image_data(k, path)
    if img is not None:
        img = cropCircle(img)
    else:
        print('couldnt crop {}'.format(k))
    cropped_images.append(img)


# folders = ['Type_1', 'Type_2', 'Type_3']
# for index,folder in enumerate(folders):
#     path = os.path.join('test', '*.jpg')
#     #path = os.path.join('test', folder, '*.jpg')
#     files=glob(path)
#     print('Started Folder:',folder )
#     for i,file in enumerate(files):
#         x_test.append(file)
#         y_test.append(index+1)

# crop, rezize and svd on images
# http://stackoverflow.com/questions/36982736/how-to-crop-biggest-rectangle-out-of-an-image
# pip install opencv-python





resiz_crop_images=np.array([img_resize(img) for img in cropped_images])

np.save("test_data.npy", resiz_crop_images)
# np.save("test_labels.npy",file['type'])