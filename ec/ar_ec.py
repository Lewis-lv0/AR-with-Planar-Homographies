import numpy as np
import cv2
import traceback
import sys
sys.path.append('../')
sys.path.append('../python/')
from python.ar import crop_margin, get_central_region, get_fps
from loadVid import loadVid
from python.opts import get_opts
from python.planarH import compositeH
import time
import multiprocessing

def process_frame(cv_cover, ar_frame, book_frame, orb, bf):
    '''
    orb: ORB detector
    bf: Brute-force matcher
    '''
    
    keypoints1, descriptors1 = orb.detectAndCompute(cv_cover, None)
    keypoints2, descriptors2 = orb.detectAndCompute(book_frame, None)
    matches = bf.match(descriptors1, descriptors2)
    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Get corresponding points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    template2photo, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    ar_frame = cv2.resize(ar_frame, (cv_cover.shape[1], cv_cover.shape[0]))
    mask = np.ones_like(ar_frame) # template mask

	#Warp mask by appropriate homography
    h, w = book_frame.shape[:2]
    warped_mask = cv2.warpPerspective(mask, template2photo, (w, h))

	#Warp template by appropriate homography
    warped_template = cv2.warpPerspective(ar_frame, template2photo, (w, h))
	# print(warped_mask.shape, warped_template.shape, img.shape)

	#Use mask to combine the warped template and the image
    composite_img = warped_template * warped_mask + book_frame * (1-warped_mask)

    return composite_img


#Write script for Q4.1x
def main(n_worker=4):
    ar_src = loadVid('../data/ar_source.mov')
    book = loadVid('../data/book.mov')
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    book_num_frames = book.shape[0]
    ar_num_frames = ar_src.shape[0]

    total_frames = ar_num_frames if ar_num_frames <= book_num_frames else book_num_frames

    # get crop information from frame 0
    cropped_ar_frame, upper_bound, lower_bound = crop_margin(ar_src[0])
    left_bound, right_bound = get_central_region(cropped_ar_frame, cv_cover)
    ar_src = ar_src[:,upper_bound:lower_bound+1,left_bound:right_bound+1,:]

    # pool = multiprocessing.Pool(n_worker)
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    all_responses = []
    # set timer
    tik = time.time()
    for i in range(total_frames):
        if i % 30 == 0:
            print(f'-- Processing frame {i} --')
        try:
            composite_img = process_frame(cv_cover, ar_src[i], book[i], orb, bf)
            all_responses.append(composite_img)
        except Exception as e:
            print(f'-- Error when processing frame {i} --')
            traceback.print_exc()
    tok = time.time()
    time_elapsed = tok - tik
    fps = total_frames / time_elapsed
    print(f'FPS is {fps}')
    video_arr = np.array(all_responses)


if __name__ == '__main__':
    main()
