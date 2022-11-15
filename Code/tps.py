import numpy as np
import cv2
import math
import sys
import copy
import cv2
import glob
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import dlib
import scipy.interpolate


def drawFacialLandmarks(img, landmarkCoord):

    jaw = landmarkCoord[0:17]
    left_ebrow = landmarkCoord[17:22]
    right_ebrow = landmarkCoord[22:27]
    nose = landmarkCoord[27:36]
    eye_left = landmarkCoord[36:42]
    eye_right = landmarkCoord[42:48]
    lips = landmarkCoord[48:]

    cv2.polylines(img, [jaw], False, (0, 255, 0), 1)
    cv2.polylines(img, [left_ebrow], False, (0, 255, 0), 1)
    cv2.polylines(img, [right_ebrow], False, (0, 255, 0), 1)
    cv2.polylines(img, [nose], False, (0, 255, 0), 1)
    cv2.polylines(img, [eye_left], False, (0, 255, 0), 1)
    cv2.polylines(img, [eye_right], False, (0, 255, 0), 1)
    cv2.polylines(img, [lips], False, (0, 255, 0), 1)

    return img

def U(r):
    U = -r**2 * (np.log(r**2))
    return np.nan_to_num(U)



def getestimatedParameters(axis, src_location,dst_location):
    p = len(dst_location)
    K = np.zeros((p,p),np.float32)
    P = np.zeros((p,3),np.float32)
    zero = np.zeros([3,3])
    #-----------------------K matrix--------------------------
    for i in range(p):
        for j in range(p):
            r = np.linalg.norm((dst_location[i,:]-dst_location[j,:]),ord =2)
            K[i,j] = U(r)

    #--------------------------P matrix------------------
    P = np.hstack((dst_location,np.ones((p,1))))

    #----------matrix [[K,P],[P.T,0]]-----------------------------------

    matrix = np.vstack((np.hstack((K,P)),np.hstack((P.transpose(),zero))))

    #--------------------matrix [[K,P],[P.T,0]]+ λI(p+3,p+3)--------------

    lamda = 1e-10
    mat = matrix + np.eye(p + 3)* lamda

    mat_inv = np.linalg.inv(mat)
    #--------------------- v = [f(x,y),0]--------------------

    v = np.concatenate((axis,[0,0,0]))

    estimated_params = np.matmul(mat_inv,v)
    return estimated_params

def blend(img1,img2,Mask):
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(Mask, kernel, iterations=1)
    r = cv2.boundingRect(Mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(np.uint8(img1), img2, mask, center, cv2.NORMAL_CLONE)

    return output



def getLandmarks(rects, image):
    img =image.copy()
    location = []
    predictor = dlib.shape_predictor('/home/naitri/Documents/733/Test/shape_predictor_68_face_landmarks.dat')
    for r, rect in enumerate(rects):
                landmarks = predictor(img, rect)

                # reshape landmarks to (68X2)
                landmarkCoord = np.zeros((68, 2), dtype='int')

                for i in range(68):
                    landmarkCoord[i] = (landmarks.part(i).x, landmarks.part(i).y)
                location.append(landmarkCoord)

                # draw bounding box on face
                cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 255), 0)

                # draw facial landmarks
                img_ = drawFacialLandmarks(img, landmarkCoord)
                
                # cv2.imshow('result', img_)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite('traingle1.jpg',img_)
    return location
def getMask(points,img):

    points = np.array(points)
    hull = cv2.convexHull(points)
 
    
    mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    cv2.fillPoly(mask, [hull], (255, 255, 255))
    mask = mask[:, :, 1]

    return mask, hull

def tps(img1,img2,points1,points2):
    detect = True
    #img1 is rambo
    #img2 is scarlett

    img2_copy = copy.deepcopy(img2)
    points1 = np.round(points1).astype(np.int32)
    points2 = np.round(points2).astype(np.int32)
    p = len(points2)
    x = points1[:,0]
    y = points1[:,1]

    est_x = getestimatedParameters(x,points1,points2)
    est_y = getestimatedParameters(y,points1,points2)

   
    a1_x = est_x[est_x.shape[0] -1]
    a2_x = est_x[est_x.shape[0] -2] 
    a3_x = est_x[est_x.shape[0] -3]
  
  
    a1_y = est_y[est_y.shape[0] -1]
    a2_y = est_y[est_y.shape[0] -2]
    a3_y = est_y[est_y.shape[0] -3]
 
 



    #generating mask
    mask,hull = getMask(points2,img2)
    r = cv2.boundingRect(np.float32(hull))

    #get all coordinates of mask 
    y_mask = np.where(mask == 255)[0].T
    x_mask= np.where(mask == 255)[1].T
    pts_mask = np.vstack((x_mask, y_mask)).T
    print(pts_mask.shape)
    pts_mask_ = np.where(mask==255)
    print(pts_mask.shape)
    cv2.imwrite('mask.jpg',mask)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


   
    #Transforming all pixels in dst img by estimated params
    for p_ in pts_mask:
        r_new = np.linalg.norm((np.array(points2)-p_),ord =2,axis=1)
        U_ = U(r_new)
        y_ = est_y[0:p].T @ U_
        x_ = est_x[0:p].T @ U_
        X = int(a1_x + a3_x*p_[0] + a2_x*p_[1] + x_)
        Y = int(a1_y+ a3_y*p_[0] + a2_y*p_[1] + y_)

        try:

            # Reading pixel value from dst image after receiving x & y coorniates and placing it onto src img
            img2[p_[1]][p_[0]] = img1[Y][X]
        except: 
            detect = False
            return detect,0
    cv2.imwrite('without_blending_tps.jpg',img2)

    # cv2.imshow('Swap without blending', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

     #Blending.

    output = blend(img2,img2_copy,mask)
    return detect,output



def thinplateSpline(img1,img2,detector):
    
    detect = True
    img1_gray= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    rects1 = detector(img1_gray, 1)
    print(rects1)
    location1 = getLandmarks(rects1, img1)
    if len(location1) != 1:
            print(" Try another.")
            detect = False

    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    rects2 = detector(img2_gray, 1)
    location2 = getLandmarks(rects2, img2)
    if len(location2) != 1:
            print(" Try another.")
            detect = False
    if detect == False:
        return detect,0
    else:
        detect, res = tps(img2,img1,location2[0],location1[0])
    # cv2.imshow('result', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return detect,res

def main():
    detector = dlib.get_frontal_face_detector()
    
    img1 = cv2.imread('input/Lisa.jpg')
    img1_gray= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    rects1 = detector(img1_gray, 1)
    location1 = getLandmarks(rects1, img1)

    img2 = cv2.imread('input/bean.jpg')
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    rects2 = detector(img2_gray, 1)
    location2 = getLandmarks(rects2, img2)

    res = tps(img2,img1,location2[0],location1[0])
    cv2.imshow('result', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






if __name__ == '__main__':
    main()
