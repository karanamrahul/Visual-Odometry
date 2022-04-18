import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from numba import jit
import time
from scipy.sparse import diags
from tqdm import tqdm

def get_points(filename1,filename2):
    """
    @brief This function is used to get the feature points from the images using ORB detector.
    
    Args:
        filename1: First image file name
        filename2: Second image file name
    
    Returns:
        points1: First image feature points
        points2: Second image feature points
    """
    # Read image - 1 ( Left image )

    print("Reading the left reference image : ", filename1)
    im1 = cv2.imread(filename1)
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    # Read Image - 2 ( Right image )
 
    print("Reading the right reference image : ", filename2)
    im2 = cv2.imread(filename2)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Finding key descriptors

    # Detect ORB features and compute descriptors.
    MAX_NUM_FEATURES = 10000
    orb = cv2.ORB_create(MAX_NUM_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    # Draw Keypoints
    im1_keypoints = cv2.drawKeypoints(im1, keypoints1, outImage=np.array([]), color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im2_keypoints = cv2.drawKeypoints(im2, keypoints2, outImage=np.array([]), color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    # Match features.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    #Sorted them based on distance , least distance signifies best match
    matches=sorted(matches,key=lambda x: x.distance)
    numgood=int(len(matches)*0.1)
    matches = matches[:numgood]

    # Drawing the good matches
    im_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imwrite("results/matches.png", im_matches)
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
  
    return points1,points2


def normalize(points):
    """
    @brief This function is used to normalize the points.
    
    Args:
        points: Points to be normalized
    
    Returns:
        points: Normalized points
    """
    points_mean = np.mean(points, axis=0)
    u_m ,v_m = points_mean[0], points_mean[1]

    u_ = points[:,0] - u_m
    v_ = points[:,1] - v_m

    s = (2/np.mean(u_**2 + v_**2))**(0.5)
    T_ = np.diag([s,s,1])
    trans = np.array([[1,0,-u_m],[0,1,-v_m],[0,0,1]])
    T = T_.dot(trans)

    x_ = np.column_stack((points, np.ones(len(points))))
    norm_pts = (T.dot(x_.T)).T

    return  norm_pts, T
    

def get_fundamental_mat(points1,points2):
    """
    @breif This function is used to get the fundamental matrix.
    
    Args: 
        points1 : First image feature points
        points2 : Second image feature points
        
    Returns:
        F : Fundamental matrix
    """
    # Normalize the points
    points1 ,T1 = normalize(points1)
    points2 ,T2 = normalize(points2)
    
    A = np.zeros((len(points1),9))
    # Construct the matrix A
    for i in range(len(points1)):
        u_l,u_r = points1[i][0],points2[i][0]
        v_l,v_r = points1[i][1],points2[i][1]
        A[i] = np.array([u_l*u_r,u_l*v_r,u_l,v_l*u_r,v_l*v_r,v_l,u_r,v_r,1])
    
    # A = np.matrix([[points1[i,0]*points2[i,0],points1[i,0]*points2[i,1],points1[i,0],points2[i,0]*points1[i,1],points1[i,1]*points2[i,1],
    #              points1[i,1],points2[i,0],points2[i,1],1] for i in range(8)])
    
    A = np.array(A)
    
    U,S,V = np.linalg.svd(A)

    F = V[-1,:].reshape(3,3)
    
    # We get the rank of F as 3 where as we need the rank of the F to be 2
    # We need to make F rank 2 by zeroing out the last singular value of F
    
    U_,S_,V_ = np.linalg.svd(F)
    S_[2] = 0
    s = np.zeros((3,3))
    for i in range(3):
        s[i,i] = S_[i]
    F = np.dot(U_,np.dot(s,V_))
    # Undo the normalization
    F_ = np.dot(T2.T, np.dot(F, T1))
    
    return F_


    
def F_RANSAC(pts1,pts2):
    """
    @breif This function is used to get the best fundamental matrix using RANSAC.
    
    Args:
        pts1 : First image feature points
        pts2 : Second image feature points
        
    Returns:
        F : Best fundamental matrix
        inliers_img1 : Inliers in the first image
        inliers_img2 : Inliers in the second image
    """
    
    points = np.concatenate((pts1,pts2),axis=1)
    max_inliners = 0
    thresh = 0.01 # Can be changed
    
    for i in range(2000):
        # Randomly select 8 points from the list of points
        point = np.random.choice(points.shape[0],8,replace=False)
        pts1_ = points[point,0:2]
        pts2_ = points[point,2:4]
        
        F_mat = get_fundamental_mat(pts1_,pts2_)
    
       
        inliers_img1 = []
        inliers_img2 = []
        
        for i in range(len(pts1)):
            x_1 ,y_1 = pts1[i][0],pts1[i][1]
            x_2 , y_2= pts2[i][0],pts2[i][1]
            
            p1_ = np.array([x_1,y_1,1])
            p2_ = np.array([x_2,y_2,1])
            dist = np.abs(np.dot(p2_.T,np.dot(F_mat,p1_)))
            
            if dist < thresh:
                inliers_img1.append(pts1[i])
                inliers_img2.append(pts2[i])
            
        n_inliers = len(inliers_img1)
        
        if n_inliers > max_inliners:
            print("Found new max inliers: ",n_inliers)
            max_inliners = n_inliers
            F_mat_ = F_mat
            inliers_img1_ = inliers_img1
            inliers_img2_ = inliers_img2
            
    return F_mat_,inliers_img1_,inliers_img2_


def get_essential_matrix(F,K):
    """
    @brief This function is used to get the essential matrix.
    
    Args:
        F : Fundamental matrix
        K : Intrinsic matrix
        
    Returns:
        E : Essential matrix
    """
    
    E = np.dot(np.dot(K.T,F),K)
    U,S,V = np.linalg.svd(E)
    S[2] = 0
    E = np.dot(U,np.dot(np.diag(S),V))
    return E
    
def get_camera_pose(E):
    """
    @brief This function is used to get the camera pose using cheirality condition.
    
    Args:
        E : Essential matrix
    
    Returns:
        R : Rotation matrix
        t : Translation vector
    
    """
    
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    Z = W.T
    R1 = np.dot(U,np.dot(W,V))
    R2 = np.dot(U,np.dot(Z,V))
    t1 = U[:,2]
    t2 = -U[:,2]
    camerapose = [[R1,t1],[R1,t2],[R2,t1],[R2,t2]]
    return camerapose

def disambiguate_camera_pose(camerapose,pts1):
    """
    @brief This function is used to disambiguate the camera pose.
    
    args:
        camerapose : List of camera poses
        pts1 : First image feature points
        
    Returns:
        R : Rotation matrix
        t : Translation vector
        
    """
    
    # We check if the camera pose is correct i.e if the pose is
    # infront of the camera or not based on cheirality test
    
    max = 0
    best_pose = 0
    for pose in camerapose:
        
        pts_front = []
        for pt in pts1:
            x=np.array([pt[0],pt[1],1])
            # We convert our image coordinates to homogenous coordinates
            # check if the point is in front of the camera or not
            v = x - pose[1]
            
            # If the product between the vector and the normal of the plane is positive
            # then the point is in front of the camera
            
            if np.dot(pose[0][2],v) > 0:
                pts_front.append(pt)
        
        if len(pts_front) > max:
            max = len(pts_front)
            best_pose = pose
        else :
            continue
    return best_pose


def img_rectification(img_l,img_r,F,pts1,pts2):
    """
    @brief This function is used to rectify the images.
    
    Args:
        img_l : Left image
        img_r : Right image
        F : Fundamental matrix
        pts1 : First image feature points
        pts2 : Second image feature points
        
    Returns:
        img_l_rect : Left rectified image
        img_r_rect : Right rectified image
        pts_rect_1 : Rectified first image feature points
        pts_rect_2 : Rectified second image feature points
        img_black_1 : Black image with rectified first image feature points
        img_black_2 : Black image with rectified second image feature points
    """
    
    img_l = cv2.imread(img_l)
    img_r = cv2.imread(img_r)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    
    # width=1920
    # height=1080
    height,width,_ = img_l.shape
    h2,w2,_ = img_r.shape
    
    # Now we calculate the Homography matrix H1 and
    # H2 which are the homographies of the left and right images respectively
    
    # We use stereoRectifyUncalibrated to get the homographies
    
    _, H1, H2= cv2.stereoRectifyUncalibrated(pts1, pts2, F, (width,height))
    
    
    print("Homography - 1 \n: ",H1)
    print("Homography - 2 \n: ",H2)
    
    # Here we rectify the ORB feature points
    pts1_rect = np.zeros((pts1.shape),dtype=np.int)
    pts2_rect = np.zeros((pts2.shape),dtype=np.int)
    for i in range(pts1.shape[0]):
        coords = np.array([pts1[i][0],pts1[i][1],1])
        pts1_new = np.dot(H1,coords)
        pts1_new[0] = int(pts1_new[0]/pts1_new[2])
        pts1_new[1] = int(pts1_new[1]/pts1_new[2])
        pts1_new = np.delete(pts1_new,2)
        pts1_rect[i] = pts1_new
        
        coords_ = np.array([pts2[i][0],pts2[i][1],1])
        pts2_new = np.dot(H2,coords_)
        pts2_new[0] = int(pts2_new[0]/pts2_new[2])
        pts2_new[1] = int(pts2_new[1]/pts2_new[2])
        pts2_new = np.delete(pts2_new,2)
        pts2_rect[i] = pts2_new
        
    
    
    # Now we use the homographies to rectify the images
    img1_rect = cv2.warpPerspective(img_l,H1,(width,height))
    img2_rect = cv2.warpPerspective(img_r,H2,(w2,h2))
    
    
    epiline_left = cv2.computeCorrespondEpilines(pts2_rect.reshape(-1,1,2),2,F).reshape(-1,3)
    epiline_right = cv2.computeCorrespondEpilines(pts1_rect.reshape(-1,1,2),1,F).reshape(-1,3)
    _, c, _ = img1_rect.shape
    colors = [tuple(random.randint(0,255) for _ in range(3)) for _ in range(8)]
    
    # Now we draw the epilines on the images
    # Create a black image just to check the epilines are correct or not
    img1_black = np.zeros((img1_rect.shape),dtype=np.uint8)
    img2_black = np.zeros((img2_rect.shape),dtype=np.uint8)
    
    
    for r, color,pt1,pt2 in zip(epiline_left, colors,pts1_rect,pts2_rect):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(img1_rect, (x0, y0), (x1, y1), color, 2)
        cv2.line(img1_black, (x0, y0), (x1, y1), color, 2)
        cv2.circle(img1_rect, (pt1[0],pt1[1]), 5, (0,0,255), 3)
        cv2.circle(img2_rect, (pt2[0],pt2[1]), 5, (255,0,0), 3)
    for r2,color,pt1,pt2 in zip(epiline_right,colors,pts1_rect,pts2_rect):
        x0, y0 = map(int, [0, -r2[2]/r2[1]])
        x1, y1 = map(int, [c, -(r2[2]+r2[0]*c)/r2[1]])
        cv2.line(img2_rect, (x0, y0), (x1, y1), color, 2)
        cv2.line(img2_black, (x0, y0), (x1, y1), color, 2)
        cv2.circle(img1_rect, (pt1[0],pt1[1]), 5, (0,0,255), 3)
        cv2.circle(img2_rect, (pt2[0],pt2[1]), 5, (255,0,0), 3)
    
    return img1_rect,img2_rect,pts1_rect,pts2_rect,img1_black,img2_black


@jit(nopython=True)
def SSD(img_left,img_right,max_disp,radius_filter):
    """
    @brief This function is used to return the sum of squared difference between two images.
    
    Args: 
        img_left : The left image used for stereo matching
        img_rigt : The right image used for stereo matching
        max_disp : Maximum disparity to be considered
        radius_filter : The filter radius considered for matching
        
    Returns:
        cost : The best matching pixels inside the cost
               volume.

    """
    
    height,width = img_left.shape
    cost = np.zeros((height,width,max_disp))
    
    for y in range(radius_filter,height-radius_filter):
        for x in range(radius_filter,width-radius_filter):
            # Now we loop over the window
            for v in range(-radius_filter,radius_filter+1):
                for u in range(-radius_filter,radius_filter+1):
                    for depth in range(max_disp):
                        cost[y,x,depth] += (img_left[y+v,x+u]-img_right[y+v,x+u-depth])**2
                        
                        
    return cost


def get_reg(max_disp,p1,p2):
    """
    @brief This function is used to return the regularization matrix.
    
    Args:
        max_disp : Maximum disparity to be considered
        p1 - Penalty when the disparity is too large
        p2 - Penalty when the disparity is too small
    
    Returns:
        reg : The regularization matrix
    """
    return np.full((max_disp,max_disp),p2) + diags([p1-p2,-p2,p1-p2],[-1,0,1],(max_disp,max_disp)).toarray()

@jit(nopython=True)
def semi_glob_cost(cost,max_disp,reg_cost):
    """
    @brief This function is used to return the best matching pixels inside the cost using semi-global matching.
    
    Args:   
        cost : The cost volume
        max_disp : Maximum disparity to be considered
        reg_cost : The regularization matrix
    
    Returns:
        cost : The best matching pixels inside the cost
        
    """
    
    height,width,max_disp = cost.shape
    estimated_cost = np.zeros((height,width,max_disp))
    for y in range(0,height):
        for x in range(0,width-1):
            for d in range(0,max_disp):
                
                total_cost = np.zeros(max_disp)
                for v in range(0,max_disp):
                    total_cost[v] = cost[y,x,v] + estimated_cost[y,x,v] + reg_cost[d,v]
                    
                 # We are choosing the minimum cost   
                estimated_cost[y,x+1,d] = np.min(total_cost)
                
                
    return estimated_cost



def semi_glob_direct_match(cost,reg_cost):
    """
    @brief This function is used to return the disparity map using semi-global matching.
    
    Args:
        cost : The cost volume
        reg_cost : The regularization matrix
        
    Returns:
        disparity_map : The disparity map
        
    """
    
    height,width,max_disp = cost.shape
    estimated_cost = np.zeros((height,width,max_disp))
    
    # Now we calulate the cost for four directions
    
    # Left to right
    estimated_cost += semi_glob_cost(cost,max_disp,reg_cost)
    
    
    # Right to left
    cost_buffer = np.zeros((height,width))
    cost_buffer = semi_glob_cost(np.flip(cost,axis=1),max_disp,reg_cost)
    estimated_cost += np.flip(cost_buffer,axis=1)
    
    
    # Top to bottom
    cost_buffer = semi_glob_cost(np.transpose(cost,(1,0,2)),max_disp,reg_cost)
    estimated_cost += np.transpose(cost_buffer,(1,0,2))
    
    # Bottom to top
    cost_buffer = semi_glob_cost(np.flip(np.transpose(cost,(1,0,2)),axis=1),max_disp,reg_cost)
    estimated_cost += np.flip(np.transpose(cost_buffer,(1,0,2)),axis=1)
    
    
    # Now we find the minimum cost
    
    disparity_map = np.zeros((height,width))
    for y in range(0,height):
        for x in range(0,width):
            disparity_map[y,x] = np.argmin(estimated_cost[y,x,:]+cost[y,x,:])
    
    return disparity_map

    
def disparity_map(cost):
    """
    @brief This function is used to return the disparity map using the simple SSD cost.
    
    Args: 
        cost : The cost volume
        max_disp : Maximum disparity to be considered
        
    Returns:
        disparity_map : The disparity map
    """
    
    height,width,_ = cost.shape
    disparity_map = np.zeros((height,width))
    
    for y in range(height):
        for x in range(width):
            disparity_map[y,x] = np.argmin(cost[y,x,:])
            
    return disparity_map


def scale_disparity_map(disparity_map):
    # This function is used to scale the disparity map to the range of 0 to 255
    
    disparity_map = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity_map


@jit(nopython=True)
def get_depth_map(disparity_map,focal_length,baseline):
    """
    @brief This function is used to return the depth map.
    
    Args: 
        disparity_map : The disparity map
        focal_length : The focal length of the camera
        baseline : The baseline of the camera
        
    Returns:
        depth_map : The depth map
    """
    
    height,width = disparity_map.shape
    depth_map = np.zeros((height,width))
    depth_array = np.zeros((height,width))
    for y in range(height):
        for x in range(width):
            if disparity_map[y,x] == 0:
                disparity_map[y,x] = np.inf
            else:
                depth_map[y,x] = 1/disparity_map[y,x]
                depth_array[y,x] = baseline*focal_length/disparity_map[y,x]
            
    return depth_map,depth_array



# This function will be used for visualization of the disparity map and depth map
def show_plots(img1,img2,img4,img_depth,scaled,n_disp,method=1):
    if method == 1:
        text = "Disparity Map with Simple block matching"
    else:
        text ="Disparity Map with Semi-Global block matching "
    fig = plt.figure(figsize = (10,10))
    plt.axis('off')
    fig.suptitle(text, fontsize=15)
    ax1 = fig.add_subplot(4,2,1)
    ax1.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)) 
    plt.axis('off')   
    ax1.set_title("Left Image (rectified)", fontsize=10)
    ax2 = fig.add_subplot(4,2,2)
    ax2.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
    plt.axis('off')     
    ax2.set_title("Right Image (rectified) (Max disparity = "+ str(n_disp), fontsize=10)
    ax3 = fig.add_subplot(4,2,3)
    ax3.imshow(img4,cmap = 'gray')
    plt.axis('off') 
    ax3.set_title("Disparity Map - Gray Scale", fontsize=10)
    ax4 = fig.add_subplot(4,2,4)
    ax4.imshow(img4 , cmap = 'viridis')
    plt.axis('off') 
    ax4.set_title("Disparity Map - Viridis Color Scale", fontsize=10)
    ax5 = fig.add_subplot(4,2,5)
    ax5.imshow(scaled,cmap='gray')
    plt.axis('off') 
    ax5.set_title('Disparity Map Scaled')
    ax6 = fig.add_subplot(4,2,6)
    ax6.imshow(img4,cmap='viridis')
    plt.axis('off') 
    ax6.set_title('Disparity Map Unscaled')
    ax7= fig.add_subplot(4,2,7)
    ax7.imshow(img_depth,cmap='gray')
    plt.axis('off') 
    ax7.set_title('Depth Map Gray')
    ax8 = fig.add_subplot(4,2,8)
    ax8.imshow(img_depth,cmap='hot')
    plt.axis('off') 
    ax8.set_title('Depth Map Color')
    plt.show()