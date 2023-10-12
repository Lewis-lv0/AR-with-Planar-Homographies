import numpy as np
import cv2
import os
import sys
import traceback

from loadVid import loadVid
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from opts import get_opts
from helper import plotMatches


def crop_margin(ar_frame):
    ''' -- crop ar frame -- '''
    # remove black margins
    gray = cv2.cvtColor(ar_frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    row_thresh = np.sum(thresh, axis=1)
    upper_bound = 0 
    lower_bound = len(row_thresh) - 1 
    for row_idx in range(len(row_thresh)):
        if row_thresh[row_idx] == 0 and upper_bound == row_idx:
            upper_bound += 1
        elif row_idx == upper_bound:
            break
    
    for row_idx in range(len(row_thresh)-1, -1, -1):
        if row_thresh[row_idx] == 0 and lower_bound == row_idx:
            lower_bound -= 1
        elif row_idx == lower_bound:
            break
    
    ar_frame = ar_frame[upper_bound:lower_bound+1]
    return ar_frame, upper_bound, lower_bound

def get_central_region(cropped_ar_frame, cv_cover):
    ar_height, ar_original_width, _ = cropped_ar_frame.shape
    cv_height, cv_width, _ = cv_cover.shape
    desired_width = ar_height * cv_width / cv_height
    left_bound = int(ar_original_width / 2 - desired_width / 2)
    right_bound = int(ar_original_width / 2 + desired_width / 2) 
    return left_bound, right_bound

def get_fps(book_vid_path):
    ''' -- get fps of book video -- '''
    cam = cv2.VideoCapture(book_vid_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    cam.release()
    return fps


def process_frame(cv_cover, ar_frame, book_frame, opts, upper_bound=None, lower_bound=None, left_bound=None, right_bound=None):
    matches, locs1, locs2 = matchPics(cv_cover, book_frame, opts)
    locs1 = locs1[matches[:,0]]
    locs2 = locs2[matches[:,1]]
    locs1[:,[0,1]], locs2[:,[0,1]] = locs1[:,[1,0]], locs2[:,[1,0]] # swap columns -> OpenCV coordinate
    bestH2to1, _ = computeH_ransac(locs1, locs2, opts)

    ar_frame = ar_frame[upper_bound:lower_bound+1,left_bound:right_bound+1,]
    ar_frame = cv2.resize(ar_frame, (cv_cover.shape[1], cv_cover.shape[0]))
    
    composite_img = compositeH(bestH2to1, ar_frame, book_frame)
    return composite_img

def main(opts, n_worker=1):
    ar_src = loadVid('../data/ar_source.mov')
    book = loadVid('../data/book.mov')
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    book_num_frames, book_height, book_width, _ = book.shape
    ar_num_frames = ar_src.shape[0]

    fps = get_fps('../data/book.mov')
    os.makedirs("../results/", exist_ok=True)
    os.makedirs('../result_img/', exist_ok=True)

    out_vid = cv2.VideoWriter("../results/ar.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (book_width, book_height))
    total_frames = ar_num_frames if ar_num_frames <= book_num_frames else book_num_frames

    # get crop information from frame 0
    cropped_ar_frame, upper_bound, lower_bound = crop_margin(ar_src[0])
    left_bound, right_bound = get_central_region(cropped_ar_frame, cv_cover)


    for i in range(total_frames):
        if i % 30 == 0:
            print(f'-- Processing frame {i} --')
        try:
            composite_img = process_frame(cv_cover, ar_src[i], book[i], opts, upper_bound, lower_bound, left_bound, right_bound)
            cv2.imwrite(f'../result_img/frame_{i}.jpg', composite_img)
        except Exception as e:
            print(f'-- Error when processing frame {i} --')
            traceback.print_exc()
    
    out_vid.release()

def make_video():
    book = loadVid('../data/book.mov')
    _, book_height, book_width, _ = book.shape
    fps = get_fps('../data/ar_source.mov')
    os.makedirs("../results/", exist_ok=True)

    out_vid = cv2.VideoWriter("../results/ar.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (book_width, book_height))
    all_out_frames = os.listdir('../result_img/')
    for i in range(len(all_out_frames)):
        composite_img = cv2.imread(os.path.join('../result_img', f'frame_{i}.jpg'))
        out_vid.write(composite_img)
    out_vid.release()


def visualize_main(opts, i, n_worker=1):
    ''' -- Method for debugging -- '''
    book = loadVid('../data/book.mov')
    cv_cover = cv2.imread('../data/cv_cover.jpg')

    matches, locs1, locs2 = matchPics(cv_cover, book[i], opts)
    plotMatches(cv_cover, book[i], matches, locs1, locs2)


if __name__ == '__main__':
    opts = get_opts()
    # main(opts)
    # visualize_main(opts, 60)
    make_video()