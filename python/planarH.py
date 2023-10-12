import numpy as np
import cv2


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	N = x1.shape[0] # number of matched points
	A = np.zeros((N*2, 9))

	for i in range(N):
		u1, v1 = x1[i,]
		u2, v2 = x2[i,]
		A[2*i,] = np.array([u2, v2, 1, 0, 0, 0, -u2*u1, -v2*u1, -u1])
		A[2*i+1,] = np.array([0, 0, 0, u2, v2, 1, -u2*v1, -v2*v1, -v1])

	_, _, Vh = np.linalg.svd(A)
	H2to1 = Vh[-1,]
	H2to1 = H2to1.reshape((3, 3))
	return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	x1_mean, x2_mean = np.mean(x1, axis=0), np.mean(x2, axis=0) 

	#Shift the origin of the points to the centroid
	x1_zero_mean, x2_zero_mean = x1 - x1_mean, x2 - x2_mean

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	max_dist_x1 = np.max(np.linalg.norm(x1_zero_mean, axis=1))  # Calculate the largest distance
	scale_factor_x1 =  np.sqrt(2) / max_dist_x1
	max_dist_x2 = np.max(np.linalg.norm(x2_zero_mean, axis=1))
	scale_factor_x2 = np.sqrt(2) / max_dist_x2

	#Similarity transform 1
	T1 = np.array([[scale_factor_x1, 0, -scale_factor_x1*x1_mean[0]], [0, scale_factor_x1, -scale_factor_x1*x1_mean[1]], [0, 0, 1]])
	x1_homo = np.hstack((x1, np.ones((x1.shape[0], 1)))) # convert to homogeneous coordinate
	x1_norm = T1 @ x1_homo.T # apply transform T1
	x1_norm =  (x1_norm[:2,] / x1_norm[2,]).T # homo -> non-homo

	#Similarity transform 2
	T2 = np.array([[scale_factor_x2, 0, -scale_factor_x2*x2_mean[0]], [0, scale_factor_x2, -scale_factor_x2*x2_mean[1]], [0, 0, 1]])
	x2_homo = np.hstack((x2, np.ones((x2.shape[0], 1))))
	x2_norm = T2 @ x2_homo.T
	x2_norm = (x2_norm[:2,] / x2_norm[2,]).T

	#Compute homography
	H_norm = computeH(x1_norm, x2_norm)

	#Denormalization
	H2to1 = np.linalg.inv(T1) @ H_norm @ T2
	return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

	num_points = locs1.shape[0]
	max_num_inlier = 0
	locs2_homo = np.hstack((locs2, np.ones((num_points, 1)))) # non-homogeneous -> homo 
	
	inliers = np.zeros((num_points,))
	bestH2to1 = None

	for _ in range(max_iters):
		sample_indices = np.random.choice(num_points, 4, replace=False)
		loc1_samples = locs1[sample_indices]
		loc2_samples = locs2[sample_indices]

		H2to1 = computeH_norm(loc1_samples, loc2_samples)
		
		loc2_2_loc1_homo = H2to1 @ locs2_homo.T
		loc2_2_loc1 = (loc2_2_loc1_homo[:2,] / loc2_2_loc1_homo[2,]).T # homo -> non-homo

		dist = np.linalg.norm(loc2_2_loc1 - locs1, axis=1) 
		is_inlier = dist < inlier_tol
		num_inliers = np.sum(is_inlier)

		if num_inliers > max_num_inlier:
			max_num_inlier = num_inliers
			inliers = is_inlier
			bestH2to1 = H2to1
	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo ---> warped img -> original img
	#For warping the template to the image, we need to invert it.
	

	#Create mask of same size as template
	mask = np.ones_like(template) # template mask

	#Warp mask by appropriate homography
	template2photo = np.linalg.inv(H2to1)
	h, w = img.shape[:2]
	warped_mask = cv2.warpPerspective(mask, template2photo, (w, h))

	#Warp template by appropriate homography
	warped_template = cv2.warpPerspective(template, template2photo, (w, h))
	# print(warped_mask.shape, warped_template.shape, img.shape)

	#Use mask to combine the warped template and the image
	composite_img = warped_template * warped_mask + img * (1-warped_mask)
	return composite_img


