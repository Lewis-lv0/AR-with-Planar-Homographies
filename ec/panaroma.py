import numpy as np
import cv2
#Import necessary functions
import sys
sys.path.append('../')
sys.path.append('../python/')
from python.opts import get_opts
from python.matchPics import matchPics
from python.planarH import computeH_ransac, compositeH



#Write script for Q4.2x
if __name__ == '__main__':
    opts = get_opts()
    left_img = cv2.imread('pano_left.jpg')
    right_img = cv2.imread('pano_right.jpg')

    lh, lw,_ = left_img.shape
    rh, rw,_ = right_img.shape

    pana = cv2.copyMakeBorder(right_img, 0, 0, lw, 0, cv2.BORDER_CONSTANT)
    # cv2.imshow('', pana)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    matches, locs1, locs2 = matchPics(left_img, pana, opts)
    locs1 = locs1[matches[:,0]]
    locs2 = locs2[matches[:,1]]
    locs1[:,[0,1]], locs2[:,[0,1]] = locs1[:,[1,0]], locs2[:,[1,0]] # swap columns

    bestH2to1, _ = computeH_ransac(locs1, locs2, opts)

    # Copy the left image onto the panorama image
    composite_img = compositeH(bestH2to1, left_img, pana)

    # remove margins on the left
    gray = cv2.cvtColor(composite_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    col_thresh = np.sum(thresh, axis=0)
    left_bound = 0 

    for col_idx in range(len(col_thresh)):
        if col_thresh[col_idx] == 0 and left_bound == col_idx:
            left_bound += 1
        elif col_idx == left_bound:
            break
    
    
    composite_img = composite_img[:,left_bound:,:]
    cv2.imwrite('./panaroma.jpg', composite_img)
    # cv2.imshow('', composite_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()