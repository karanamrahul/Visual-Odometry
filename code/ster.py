

"""
Visual Odometry
@author Rahul Karanam
@ date: 16/04/2022
@brief This project explains how to implement a visual odometry for a stereo camera system.
       Stereo Matching of the images is done using Semi Global Block Matching.
"""
from helper_ import *

K_mat_curule = np.array([1758.23 ,0 ,977.42, 0 ,1758.23 ,552.15, 0, 0 ,1]).reshape(3,3)
K_mat_octagon = np.array([1742.11, 0, 804.90, 0 ,1742.11, 541.22, 0, 0, 1]).reshape(3,3)
K_mat_pendulum = np.array([1729.05 ,0,-364.24, 0 ,1729.05 ,552.22, 0, 0, 1]).reshape(3,3)
Focal_curule = 1758.23
Focal_octagon = 1742.11
Focal_pendulum = 1729.05
baseline_curule = 88.39
baseline_octagon = 221.76
baseline_pendulum = 537.75

def stero_vision():
    print("Please choose the dataset number (1) - Curule (2) - Octagon (3) - Pendulum : \n")
    flag = int(input())
    print("Please choose the method number (1) - Simple Block Matching  (2) - Semi Global Block Matching : \n")
    method = int(input())
    
    if flag == 1:
        K = K_mat_curule
        filename1="data/curule/im0.png"
        filename2="data/curule/im1.png"
        focal = Focal_curule
        baseline = baseline_curule
        n_disp = 220
    elif flag == 2:
        K = K_mat_octagon
        filename1="data/octagon/im0.png"
        filename2="data/octagon/im1.png"
        focal = Focal_octagon
        baseline = baseline_octagon
        n_disp = 100
    elif flag == 3:
        K = K_mat_pendulum
        filename1="data/pendulum/im0.png"
        filename2="data/pendulum/im1.png"
        focal = Focal_pendulum
        baseline = baseline_pendulum
        n_disp = 180
    else : 
        print("Invalid flag")
        
    ########## Calibration ##########  
    # We get the features and descriptors of the images using ORB detector and Brute Force matcher
    points1, points2 = get_points(filename1,filename2)
    
    # Compute the fundamental matrix using RANSAC and the normalized points
    F,pts_1,pts_2 = F_RANSAC(points1,points2)
    
    E = get_essential_matrix(F,K)
    

    camera_pose = get_camera_pose(E)
    best_pose = disambiguate_camera_pose(camera_pose,pts_1)

    ####### Stereo Rectification #######
    img1,img2,pts1_rect,pts2_rect,im3,im4 = img_rectification(filename1,filename2,F,pts_1,pts_2)
    
    
    # # cv2.imwrite("data/curule/rectified_im0.png",img1)
    # # cv2.imwrite("data/curule/rectified_im1.png",img2)
    
    
    ##########  Stereo Matching and Disparity Map ##########
    img_left = cv2.imread(filename1,0)
    img_right = cv2.imread(filename2,0)
    start = time.time()
    
    # Calculating the Cost using Sum of Squared Differences
    cost = SSD(img_left,img_right,n_disp,4)
    
    # Calculating the disparity map using Simple Block Matching 
    img4 = disparity_map(cost)
    
    
    # Applying the regularization to the disparity map for the  Semi Global Block Matching
    reg_cost = get_reg(n_disp,0.025,0.5)
    
    # Uncomment the below line show the disparity map using Semi Global Block Matching.
    # img4=semi_glob_direct_match(cost,reg_cost)
    
    # Scaled disparity map
    scaled_disp_map = scale_disparity_map(img4)
    
    
    
    ######## Depth Map ########
    img_depth,depth_arr = get_depth_map(img4,focal,baseline)
    
   
   
    end= time.time()
    print("Time : ",end-start)
    
    
    ######### Visualization #########
    
    show_plots(img1,img2,img4,img_depth,scaled_disp_map,n_disp,method)
    
  
  
    ###### Output ######
    print("Fundamental Matrix : \n", F)
    print("Rank of F : ", np.linalg.matrix_rank(F))
    print("Essential Matrix : \n", E)
    print("Best pose: Rotation\n ",best_pose[0])
    print("Best pose: Translation \n ",best_pose[1])
    
    

    
if __name__ == "__main__":
    stero_vision()
    