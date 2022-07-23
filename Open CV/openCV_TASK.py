import cv2
import numpy as np
import cv2.aruco as aruco
import math

#function for calculating corners & ids of aruco_markers:
def Findaruco(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                   
    key = getattr(aruco, f'DICT_5X5_250')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create() 

    (corners, ids, rejected) = aruco.detectMarkers(img, arucoDict, parameters = arucoParam)
    cv2.imwrite(f"{int(ids)}.jpg",img)                   # saving the aruco img with their id name.
    return (corners, ids, rejected) 

# fuction for converting corners & ids type into integer: 
def int_corners_ids(corners, ids):
    for(markercorner, markerid) in zip(corners, ids):
        corners = markercorner.reshape((4,2))
        (topleft, topright, bottomright, bottomleft) = corners

        topleft = (int(topleft[0]), int(topleft[1]))
        topright = (int(topright[0]), int(topright[1]))
        bottomright = (int(bottomright[0]), int(bottomright[1]))
        bottomleft = (int(bottomleft[0]), int(bottomleft[1]))
    
# (cx, cy) is the center of the aruco:
        cx = int((bottomleft[0] + topright[0])/2.0)
        cy = int((bottomleft[1] + topright[1])/2.0)

# (mx, my) is the mid point of left side to the centre:
        mx = int((bottomleft[0] + topleft[0])/2.0)
        my = int((bottomleft[1] + topleft[1])/2.0)

        val = ((my-cy)/(mx-cx))

# angle by which img is rotated into clockwise:
        angle = math.degrees(math.atan(val))
    
    ids = int(ids)
    
    return (corners, ids, angle)

# function for calculating countoursDICT with ids of image:
def mask(img, color):
    colorarr = color.values()
    colorids = color.keys()
    color_mask = {}
    mask_contours = {}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                       # img converted into hsv from bgr.

# for loop for finding mask of the different color:
    for (color, ids) in zip(colorarr, colorids):
        mask = cv2.inRange(hsv, color[0], color[1])

    # using medianBlur remove noise of the img:    
        mask = cv2.medianBlur(mask, 5)  
        color_mask[ids] = mask  

# for loop for finding coutour of the img:
    for (img, ids) in zip(color_mask.values(), color_mask.keys()):
        cont, hier = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # for loop for using approxPolyDP by which contours makes perfect:     
        for cnt in cont:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            mask_contours[ids] = approx
      
    return (mask_contours)

# function for rotation and slicing of aruco_markers:
def remove_extra_padding(img, angle):
    h,w = img.shape[:-1]
    
    # rotation point (centre) of the img;
    rot_point = w//2, h//2

    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)          # ing converted into HSV from color.
    rot_mat = cv2.getRotationMatrix2D(rot_point,angle,1)   # gives a matrix in which rotation in anticlockwise with given angle about rot_point.    
    rot = cv2.warpAffine(img_hsv,rot_mat,(w,h))            # rotate the hsv_img using rot_matrix.
    img2 = cv2.cvtColor(rot, cv2.COLOR_HSV2BGR)            # now, img converted into bgr from hsv.

# find out corners, ids of the img:
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_5X5_250')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create() 

    (corners, ids, rejected) = aruco.detectMarkers(img2, arucoDict, parameters = arucoParam)

# converting corners & ids type into integer: 
    for corner in corners:    
        corner = corner.reshape((4,2))
        (topleft, topright, bottomright, bottomleft) = corner

        topleft = (int(topleft[0]), int(topleft[1]))
        topright = (int(topright[0]), int(topright[1]))
        bottomright = (int(bottomright[0]), int(bottomright[1]))
        bottomleft = (int(bottomleft[0]), int(bottomleft[1]))

# slicing of img(remove the extra padding):
    crop_img = img2[topleft[0]:topright[0], topleft[1]:bottomleft[1]]

# create a numpy of corners of the img:    
    crop_cr = np.array([[0,0], [crop_img.shape[1],0], [crop_img.shape[1],crop_img.shape[0]], [0,crop_img.shape[0]]])
    
    return (crop_img, crop_cr)


# import CVtask image and resize it:
img = cv2.imread("Atulya_Opencv\CVtask.jpg")
img = cv2.resize(img, None, fx=0.5, fy=0.5)

# showing CVtsk img. 
cv2.imshow("CVtask", img)                                              

# HSV value range of following color from built mask library using trackbar:
green = np.array([[32,32,150], [60,255,255]])
orange = np.array([[8,180,14], [100,255,255]])
black= np.array([[0,0,0], [0,0,0]])
pinkpeach = np.array([[0,15,0], [60,29,239]])

color ={1:green, 2:orange, 3:black, 4:pinkpeach}       # DICT of ids and color, ids value aruco fill in that color.
mask_contours = mask(img, color)                  # mask_contours are DICT of contour with their ids w.r.t color.

# import aruco markers:
aruco_marker1 = cv2.imread("Atulya_Opencv/Ha.jpg")
aruco_marker2 = cv2.imread("Atulya_Opencv/HaHa.jpg")
aruco_marker3 = cv2.imread("Atulya_Opencv/LMAO.jpg")
aruco_marker4 = cv2.imread("Atulya_Opencv/XD.jpg")

# find out corners, ids of aruco markers:
(corners1, ids1, rejected1) = Findaruco(aruco_marker1)
(corners2, ids2, rejected2) = Findaruco(aruco_marker2)
(corners3, ids3, rejected3) = Findaruco(aruco_marker3)
(corners4, ids4, rejected4) = Findaruco(aruco_marker4)

# convert into integer values and return also angle by which img tilted:
corners1, ids1, angle1 = int_corners_ids(corners1, ids1)
corners2, ids2, angle2 = int_corners_ids(corners2, ids2)
corners3, ids3, angle3 = int_corners_ids(corners3, ids3)
corners4, ids4, angle4 = int_corners_ids(corners4, ids4)
corners = np.array([corners1, corners2, corners3, corners4])          # making a np array of all corners.

# Making a arucoDICT of markers with their ids value:
aruco_dict = {ids1:[angle1, aruco_marker1], ids2:[angle2, aruco_marker2], ids3:[angle3,aruco_marker3], ids4:[angle4, aruco_marker4]}

approx = {}                                                           # creating a empty DICT:

# for loop for containing only square countour into approxDICT with their ids:
for (ids) in (mask_contours):
    x,y,w,h = cv2.boundingRect(mask_contours[ids])

    #aspect_ratio is the ratio of adjacent sides:
    aspect_ratio = float(w)/h
    if ((0.95 <= aspect_ratio) & (aspect_ratio <= 1.05)):
        approx[ids] = mask_contours[ids]   


# for loop for updating aruco img(remove outsides white space) and pasting into CVtask img:
for ids in aruco_dict:
    pt1 = approx[ids]
    angle = aruco_dict[ids][0]
    aruco_img = aruco_dict[ids][1]

    if(ids == 1):
        aruco_img, pt2 = remove_extra_padding(aruco_img, angle)

    elif(ids == 2):
        aruco_img, pt2 = remove_extra_padding(aruco_img, angle)

    elif(ids == 3):
        aruco_img, pt2 = remove_extra_padding(aruco_img, angle)  
    
    elif(ids == 4):
        aruco_img, pt2 = remove_extra_padding(aruco_img, angle)

    matrix, _ = cv2.findHomography(pt2, pt1) 
    warp_img =cv2.warpPerspective(aruco_img, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pt1, (0,0,0))
    img = img + warp_img

# showing final img:    
cv2.imshow("final", img)
cv2.imwrite("final.jpg",img)                        # saving the final img.
cv2.waitKey(0)
cv2.destroyAllWindows()      
    









