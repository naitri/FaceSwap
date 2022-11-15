import cv2
import numpy as np
import dlib
from scipy.interpolate import interp2d

"""
    Input: BGR image, path to dlib pre-trained shape predictor
    Output: Draws a rectangle for face detection and 68 detected facial landmarks per face detected
            Returns x, y coordinates of facial landmarks detected

    References: https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
                https://github.com/davisking/dlib/blob/master/python_examples/face_landmark_detection.py
"""
def get_facial_landmarks(img, predictor_path):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # The second argument means that we upsample the image 1 time
    rects = detector(gray_img, 1)
    num_faces = len(rects)
    print("Number of faces detected: {}".format(num_faces))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = []
    landmarks = []
    points = []

    for (i, rect) in enumerate(rects):
        points = []

        shape = predictor(gray_img, rect)
        landmarks = np.zeros((68, 2), dtype=int)

        for i in range(68):
            landmarks[i] = (shape.part(i).x, shape.part(i).y)

        x = rect.left()
        y = rect.top()
        w, h = rect.right() - x, rect.bottom() - y

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for (x, y) in landmarks:
           cv2.circle(img, (x, y), 5, (0, 0, 255), 2)
           points.append((int(x), int(y)))

        if num_faces > 1:
            faces.append(points)

    if num_faces > 1:
        return num_faces, faces
    else:
        return num_faces, points

"""
    Input: triangle coordinates, boolean flag whether to invert matrix
    Output: Barycentric transformation matrix
"""
def compute_barycentric_matrix(triangle, inv_flag):
    matrix = [[triangle[0][0], triangle[1][0], triangle[2][0]],
              [triangle[0][1], triangle[1][1], triangle[2][1]],
              [1,1,1]]

    if inv_flag:
        matrix = np.linalg.inv(matrix)

    return matrix

"""
    - Computes barycentric transformation.
    - Identifies the corresponding pixels in the source image.
    - Copies back the pixel values of the identified source locations to the target image
    Reference: 
"""
def warp_triangulation(img_src, triangle_src, triangle_dest, size, input_type):
    epsilon = 0.1
    rect = cv2.boundingRect(np.float32([triangle_dest]))

    x_left = rect[0]
    x_right = rect[0] + rect[2]
    y_top = rect[1]
    y_bottom = rect[1] + rect[3]

    matrix_dest = compute_barycentric_matrix(triangle_dest, True)

    grid = np.mgrid[x_left:x_right, y_top:y_bottom].reshape(2,-1)
    grid = np.vstack((grid, np.ones((1, grid.shape[1]))))
    barycoords = np.dot(matrix_dest, grid)
    
    triangle_dest = []
    b = np.all(barycoords > - epsilon, axis=0)
    a = np.all(barycoords < 1 + epsilon, axis=0)

    for i in range(len(a)):
        triangle_dest.append(a[i] and b[i])
    y_dest = []
    x_dest = []
    for i in range(len(triangle_dest)):
        if(triangle_dest[i]):
            y_dest.append(i % rect[3])
            x_dest.append(i / rect[3])

    barycoords = barycoords[:, np.all(-epsilon < barycoords, axis=0)]
    barycoords = barycoords[:, np.all(barycoords < 1 + epsilon, axis=0)]
  
    matrix_src = compute_barycentric_matrix(triangle_src, False)
    pts = np.matmul(matrix_src, barycoords)
    
    xA = pts[0,:] / pts[2,:]
    yA = pts[1,:] / pts[2,:]

    dest = np.zeros((size[1],size[0],3), np.uint8)

    xs = np.linspace(0, img_src.shape[1], num=img_src.shape[1], endpoint=False)
    ys = np.linspace(0, img_src.shape[0], num=img_src.shape[0], endpoint=False)

    # workaround for occasions where the number of data points in the matrix for interpolation are less than the minimum allowed number.
    if input_type == 1:
        interp_type = 'cubic'
        lim = 3
    else:
        interp_type = 'linear'
        lim = 1

    flag = True
    if img_src[:, :, 0].shape[0] > lim and img_src[:, :, 0].shape[1] > lim:
        b = img_src[:, :, 0]
        fb = interp2d(xs, ys, b, kind=interp_type)
    else:
        flag = False

    if img_src[:, :, 1].shape[0] > lim and img_src[:, :, 1].shape[1] > lim:
        g = img_src[:, :, 1]
        fg = interp2d(xs, ys, g, kind=interp_type)
    else:
        flag = False

    if img_src[:, :, 2].shape[0] > lim and img_src[:, :, 2].shape[1] > lim:
        r = img_src[:, :, 2]
        fr = interp2d(xs, ys, r, kind=interp_type)
    else:
        flag = False

    if flag:
        for i, (x,y) in enumerate(zip(xA.flat, yA.flat)):

            blue = fb(x, y)[0]
            green = fg(x, y)[0]
            red = fr(x, y)[0]
            dest[int(y_dest[i]), int(x_dest[i])] = (blue,green,red)

    return dest

"""
    Checks if a point is inside a rectangle.
    Reference: https://learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/
"""
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True

"""
    Computes and returns Delaunay triangles
    Reference: https://learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/
"""
def get_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p) 
    
    triangleList = subdiv.getTriangleList();
    
    delaunayTri = []
    pt = []
    for t in triangleList:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))    
        pt = []
    return delaunayTri

"""
    - Applies affine transformation/delaunay triangulation
    - Applies alpha blending to triangles from source and destination images to output image
"""
def warp_triangle(img_src, img_dest, triangle1, triangle2, input_type) :

    # Find bounding rectangle for each triangle
    rect_src = cv2.boundingRect(np.float32([triangle1]))
    rect_dest = cv2.boundingRect(np.float32([triangle2]))

    # Offset points by left top corner
    t1Rect = [] 
    t2Rect = []

    for i in range(0, 3):
        t1Rect.append(((triangle1[i][0] - rect_src[0]),(triangle1[i][1] - rect_src[1])))
        t2Rect.append(((triangle2[i][0] - rect_dest[0]),(triangle2[i][1] - rect_dest[1])))


    # Create mask
    mask = np.zeros((rect_dest[3], rect_dest[2], 3), dtype = np.float32)

    cv2.fillConvexPoly(mask, np.int32(t2Rect), (1.0, 1.0, 1.0), 16, 0);

    # Warp small patches
    img1Rect = img_src[rect_src[1]:rect_src[1] + rect_src[3], rect_src[0]:rect_src[0] + rect_src[2]]
    img2Rect = np.zeros((rect_dest[3], rect_dest[2]), dtype = img1Rect.dtype)
    
    size = (rect_dest[2], rect_dest[3])

    img2Rect = warp_triangulation(img1Rect, t1Rect, t2Rect, size, input_type)
    img2Rect = img2Rect * mask

    a = (1.0, 1.0, 1.0) - mask

    # Copy triangular patch region to the output image
    img_dest[rect_dest[1]:rect_dest[1]+rect_dest[3], rect_dest[0]:rect_dest[0]+rect_dest[2]] = img_dest[rect_dest[1]:rect_dest[1]+rect_dest[3], rect_dest[0]:rect_dest[0]+rect_dest[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img_dest[rect_dest[1]:rect_dest[1]+rect_dest[3], rect_dest[0]:rect_dest[0]+rect_dest[2]] = img_dest[rect_dest[1]:rect_dest[1]+rect_dest[3], rect_dest[0]:rect_dest[0]+rect_dest[2]] + img2Rect

"""
    Applies Delaunay triangulation using the points extracted from the computed convex hull.
"""
def triangulation(img_src, img_dest, img_warped, hull_src, hull_dest, input_type):
    size_dest = img_dest.shape    
    rect = (0, 0, size_dest[1], size_dest[0])
    triangle_list = get_triangles(rect, hull_dest)

    if len(triangle_list) == 0:
        quit()

    # Apply an affine transformation to the Delaunay triangles
    for i in range(0, len(triangle_list)):
        triangle1 = []
        triangle2 = []
        
        # Get points for source and destination images, corresponding to the triangles
        for j in range(0, 3):
            triangle1.append(hull_src[triangle_list[i][j]])
            triangle2.append(hull_dest[triangle_list[i][j]])
        
        warp_triangle(img_src, img_warped, triangle1, triangle2, input_type)

    return img_warped