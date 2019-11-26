import os
import cv2
import numpy as np

#director to all the frame pngs
data_dir = "/mnt/data/datasets/kitti/dataset/sequences/00/image_0/"

#TODO find actual camera matrix
cam_matrix = np.zeros((3,3), dtype=float)
cam_matrix[0,0] = 1
cam_matrix[0,2] = 1
cam_matrix[1,2] = 1
cam_matrix[1,1] = 1
cam_matrix[2,2] = 1

def flann_ratio_match(flann, des1, des2):
    matches = flann.knnMatch(des1, des2, k=2)
    #perform lowe's ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    return good

def retrieve_trans(good_m, kp1, kp2):
    #translate matches to pixel cordinates... i thinl
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_m ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_m ]).reshape(-1,1,2)
    #find transformation and remve points that dont fit.
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, cam_matrix, cv2.RANSAC, 0.999, 0.5)
    src_pts = src_pts[mask.ravel() == 1]
    dst_pts = dst_pts[mask.ravel() == 1]

    _, _, T = cv2.decomposeEssentialMat(E, src_pts, dst_pts)
    return T.reshape(1,3)[0]
    
if __name__ == '__main__':
    sift = cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 6)
    search_params = dict(checks = 150)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    total_trans = None
    #iterate every frame in data_dir
    for image_id in range(1,4541,5):
        frame = cv2.imread(os.path.join(data_dir, str(image_id).zfill(6) + '.png'))
        #todo adding caching for prev_frame and its keypoints and descriptors
        prev_frame = cv2.imread(os.path.join(data_dir, str(image_id-1).zfill(6) + '.png'))
        print("idx: ", image_id)
        
        kp1, des1 = sift.detectAndCompute(prev_frame, None)
        kp2, des2  = sift.detectAndCompute(frame, None)

        good_m = flann_ratio_match(flann, des1,des2)
        T = retrieve_trans(good_m, kp1, kp2)
        if total_trans is None:
            total_trans = T
        else:
            total_trans = np.add(total_trans, T)
#        print("trans step\n", T)
        print("trans total\n", total_trans)
        cv2.imshow("frame", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break

