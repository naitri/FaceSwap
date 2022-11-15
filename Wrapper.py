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
import argparse
from Code.tps import *
from PRNet.main import *


from Code.delaunay import get_facial_landmarks, triangulation

"""
    Applies Poisson blending to an image.
    Reference: https://learnopencv.com/seamless-cloning-using-opencv-python-cpp/
"""
def blend(hull_dest, img_dest, img_warped):
    hull = []

    for point in hull_dest:
        hull.append((point[0], point[1]))

    mask = np.zeros(img_dest.shape, dtype=img_dest.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull), (255, 255, 255))
    rect = cv2.boundingRect(np.float32([hull_dest]))
    center = ((rect[0] + int(rect[2] / 2), rect[1] + int(rect[3] / 2)))

    result = cv2.seamlessClone(np.uint8(img_warped), img_dest, mask, center, cv2.NORMAL_CLONE)
    return result

"""
    - Computes convex hull of source and destination images.
    - Calls functions to perform triangulation and warping.
"""
def triangulation_warping(img_src, img_dest, landmarks_src, landmarks_dest, input_type):
    hull_src = []
    hull_dest = []
    img_warped = img_dest.copy()

    hull = cv2.convexHull(np.array(landmarks_dest), returnPoints=False)

    for i in range(len(hull)):
        hull_src.append(landmarks_src[int(hull[i])])
        hull_dest.append(landmarks_dest[int(hull[i])])

    img_warped = triangulation(img_src, img_dest, img_warped, hull_src, hull_dest, input_type)
    result = blend(hull_dest, img_dest, img_warped)

    return result

"""
    Handles arguments and calls functions for faceswap.
    References: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
"""

def main():
    parser = argparse.ArgumentParser(description='FaceSwap')
    parser.add_argument('--src', required=False, help='Path for source image')
    parser.add_argument('--dst', required=False, help='Path for reference image')
    parser.add_argument('--result', required=False, help='Path for storing output images/video')
    parser.add_argument('--video', required=False, help='Path for input video')
    parser.add_argument('--method', default="tps",  help='Choose method 1) Delaunay Traingulation : deltriangle 2) Thinplate spline : tps 3) PRNet :prnet')
    parser.add_argument('--mode',default="1",  help='Choose Mode 1: Swap faces in two images, 2: Swap face in video with an image, 3: Swap faces in a video')
    args = parser.parse_args()

    mode = args.mode
    video = args.video
    src = args.src
    dst = args.dst
    result = args.result
    method = args.method


    #-------------------------------------MODE 1: Swap faces in two images-------------------------
    if mode == "1":
        img1 = cv2.imread(src)
        img2 = cv2.imread(dst)
        
        if method == "tps":
            print("Face Swapping Using Thin Plate spline method")
            detector = dlib.get_frontal_face_detector()
            detect,result = thinplateSpline(img1,img2,detector)
            # cv2.imshow('result using tps', result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite('result_using_tps.png', result)


        elif method == "deltriangle":
            print("Face Swapping Using Thin Plate spline method")

            num_faces, points1 = get_facial_landmarks(img1, "./Data/shape_predictor_68_face_landmarks.dat")

            if(num_faces != 1):
                print("No face detected, or more than one face were detected.")
                exit()

            num_faces, points2 = get_facial_landmarks(img2, predictor_path)

            if(num_faces != 1):
                print("No face detected, or more than one face were detected.")
                exit()
            output = triangulation_warping(img1, img2, points1, points2, 1)
            cv2.imwrite('result_using_del.png', output)

        elif method == "prnet":

            print("Face Swapping Using PRNet method")
            result = prnet(img1,img2)
            # cv2.imshow('result using prnet', result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite('result_using_prnet.png', result)
        else:
            print("Incorrect method chosen")

    #-------------------------------------MODE 2: Swap face in video with an image-------------------------
    elif mode == "2":
        img2 = cv2.imread(src)
        num_faces, points2 = get_facial_landmarks(img2, "./Data/shape_predictor_68_face_landmarks.dat")

        if(num_faces != 1):
            print("No face detected, or more than one face were detected.")
            exit()
        cap = cv2.VideoCapture(video)
        ret,img = cap.read()
        height, width = img.shape[0], img.shape[1]


        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width,height))
        detector = dlib.get_frontal_face_detector()


        while (cap.isOpened()):

            ret, img1 = cap.read()
            if ret == True:
               

                if method == "tps":
                    print("Face Swapping Using Thin Plate spline method")
                    detect,result = thinplateSpline(img1,img2,detector)
                    if detect == False:
                        continue
                    else:
                        # cv2.imshow('result using tps', result)

                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        cv2.imwrite('tps_mode2.jpg',result)
                        out.write(result)


                elif method == "deltriangle":
                    print("Face Swapping Using Thin Plate spline method")

                    num_faces, points1 = get_facial_landmarks(img1, "./Data/shape_predictor_68_face_landmarks.dat")
                    if(num_faces == 0):
                        continue

                    output = triangulation_warping(img2, img1, points2, points1, 1)
                    out.write(output)


                elif method == "prnet":

                    print("Face Swapping Using PRNet method")
                    try:

                        result = prnet(img1,img2)
                        out.write(result)
                        cv2.imwrite('result_using_prnet.jpg', result)
                        # cv2.imshow('result using prnet', result)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                    except:
                        continue
                else:
                    print("Incorrect method chosen")

            else:
                exit()


    #-------------------------------------MODE 3:Swap faces in a video-------------------------
    elif mode == "3":
        cap = cv2.VideoCapture(video)
        ret,img = cap.read()

        height, width = img.shape[0], img.shape[1]

        # Define the codec and create VideoWriter object (MJPG is compatible with .mp4)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter('{}_{}_result.avi'.format(method, video), fourcc, 15, (width, height))

        detector = dlib.get_frontal_face_detector()
        while (cap.isOpened()):

            ret, img = cap.read()
            if ret == True:
                

                if method == "tps":
                    print("Face Swapping Using Thin Plate spline method")

                    num_faces,points = get_facial_landmarks(img, "./Data/shape_predictor_68_face_landmarks.dat")

                    if(num_faces!=2):
                        print("{} faces detected. Moving to the next frame.".format(num_faces))
                        continue
                    else:
                        points1 = points[0]
                        points2 = points[1]
                        result_temp = tps(img,img,points1, points2)
                        result = tps(result_temp,img,points2, points1)
                        cv2.imshow('result using tps', result)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        # out.write(result)


                elif method == "deltriangle":
                    print("Face Swapping Using Thin Plate spline method")

                    num_faces,points = get_facial_landmarks(img, "./Data/shape_predictor_68_face_landmarks.dat")

                    if(num_faces!=2):
                        print("{} faces detected. Moving to the next frame.".format(num_faces))
                        continue
                    else:
                        points1 = points[0]
                        points2 = points[1]

                    temp = triangulation_warping(img, img, points1, points2, 2)
                    output = triangulation_warping(img, temp, points2, points1, 2)
                    out.write(output)


                elif method == "prnet":

                    print("Face Swapping Using PRNet method")
                    result = prnet(frame,frame)
                    out.write(result)
                    # cv2.imshow('result using prnet', result)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                else:
                    print("Incorrect method chosen")


            else:
                exit()

    else:
        print("oops")



if __name__ == '__main__':
    main()





