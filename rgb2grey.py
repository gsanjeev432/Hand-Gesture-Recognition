import cv2
import os

def rgb2gray(img):
    
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, 25, 255,
                                   cv2.THRESH_BINARY)
    return thresh

train_color_img_dir = "D:\\Sanjeev\\Hand Gesture Recognition\\ASL\\training\\"
train_grey_img_dir = "D:\\Sanjeev\\Hand Gesture Recognition\\grey_train\\"

train_color_img_files = os.listdir(train_color_img_dir)
train_grey_img_files = os.listdir(train_grey_img_dir)

for train_color_img_file in train_color_img_files:
    train_color = os.listdir(train_color_img_dir+train_color_img_file)
    for p in train_color:
        img = cv2.imread(train_color_img_dir+train_color_img_file+"\\"+p)
        res = cv2.resize(img, (100,100),interpolation = cv2.INTER_AREA)
        output_image = rgb2gray(res)
        
        cv2.imwrite(train_grey_img_dir+train_color_img_file+"\\"+p,output_image)
        

test_color_img_dir = "D:\\Sanjeev\\Hand Gesture Recognition\\ASL\\testing\\"
test_grey_img_dir = "D:\\Sanjeev\\Hand Gesture Recognition\\grey_test\\"

test_color_img_files = os.listdir(test_color_img_dir)
test_grey_img_files = os.listdir(test_grey_img_dir)

for test_color_img_file in test_color_img_files:
    test_color = os.listdir(test_color_img_dir+test_color_img_file)
    for q in test_color:
        img = cv2.imread(test_color_img_dir+test_color_img_file+"\\"+q)
        res = cv2.resize(img, (100,100),interpolation = cv2.INTER_AREA)
        output_image = rgb2gray(res)
        
        cv2.imwrite(test_grey_img_dir+test_color_img_file+"\\"+q,output_image)
    
