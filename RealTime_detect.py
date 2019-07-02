import cv2
from dtaidistance import dtw
import numpy as np

def feat(path):
    img = path
    
    from skimage.feature import corner_harris
    
    coords = corner_harris(img)
    
    feat = coords.flatten()
    
    return feat

def segment(image, threshold=140):

    # threshold the diff image so that we get the foreground
    __, thresholded = cv2.threshold(image,
                                threshold,
                                255,
                                cv2.THRESH_BINARY_INV)
    return thresholded
    
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
top, right, bottom, left = 10, 10, 200, 180

x_train_file = np.load('new_train1_data.npy')
y_train_file = np.load('new_train1_label.npy')

x_train = np.array(x_train_file)
y_train = np.array(y_train_file)

while True:
    ret, frame = cam.read()
    
    clone = frame.copy()

    # get the height and width of the frame
    (height, width) = frame.shape[:2]

    # get the ROI
    roi = frame[top:bottom, right:left]
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        print("Please wait")
        
        res = cv2.resize(gray, (100,100),interpolation = cv2.INTER_AREA)
        hand = segment(res)

            # check whether hand region is segmented
        if hand is not None:
            thresholded = hand
            thresholded = cv2.erode(thresholded, None, iterations=2)
            thresholded = cv2.dilate(thresholded, None, iterations=2)
            cv2.imshow("Thesholded", thresholded)
            dtw_feat = feat(thresholded)
            final = []
            for xtrain in x_train:
                distance = dtw.distance_fast(dtw_feat,xtrain)
                final.append(distance)
            mini = final.index(min(final))
            pred = y_train[mini]
            print("Predicted:"+pred)
        
            cv2.putText(clone,pred, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 127,3)
            
            cv2.imwrite(img_name, clone)
            print("{} written!".format(img_name))
        img_counter += 1
    
    
        
    cv2.imshow("Video Feed", clone)
    
    cv2.imshow("test", frame)
    
        

cam.release()

cv2.destroyAllWindows()