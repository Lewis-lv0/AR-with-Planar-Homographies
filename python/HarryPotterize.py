import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

def warp(opts):

    img1 = cv2.imread('../data/cv_cover.jpg')
    img2 = cv2.imread('../data/cv_desk.png')
    img3 = cv2.imread('../data/hp_cover.jpg')
    matches, locs1, locs2 = matchPics(img1, img2, opts) # cover to desk

    locs1 = locs1[matches[:,0]]
    locs2 = locs2[matches[:,1]]
    locs1[:,[0,1]], locs2[:,[0,1]] = locs1[:,[1,0]], locs2[:,[1,0]] # swap columns

    bestH2to1, inliers = computeH_ransac(locs1, locs2, opts)

    img3 = cv2.resize(img3, (img1.shape[1], img1.shape[0]))
    warp_hp_cover = compositeH(bestH2to1, img3, img2)
    cv2.imwrite('../warp_hp_cover_1000_2.png', warp_hp_cover)
    # cv2.imshow('', warp_hp_cover)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    '''
    height, width = img2.shape[:2] # shape of cv desk
    warp_hp_cover = cv2.warpPerspective(img3, bestH2to1, (width, height))
    cv2.imshow(' ', warp_hp_cover)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    

#Write script for Q2.2.4
opts = get_opts()
warp(opts)