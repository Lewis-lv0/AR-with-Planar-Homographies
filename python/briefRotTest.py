import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts
import scipy.ndimage
from matplotlib import pyplot as plt
import pickle


opts = get_opts()
'''
#Q2.1.6
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')

num_match_list = []
for i in range(36):
	#Rotate Image
	cv_rotated = scipy.ndimage.rotate(cv_cover, i*10)
	#Compute features, descriptors and Match features
	matches, _, _ = matchPics(cv_cover, cv_rotated, opts)
	#Update histogram
	num_match_list.append(len(matches))
	# pass # comment out when code is ready
with open('q_2_1_6.pkl', 'wb') as file:
    pickle.dump(num_match_list, file)


with open('q_2_1_6.pkl', 'rb') as file:
    num_match_list = pickle.load(file)
print(len(num_match_list))

plt.bar(np.arange(start=0, stop=360, step=10), num_match_list)

plt.xlabel('rotation')
plt.ylabel('number of matches')
plt.savefig('q_2_1_6.png')

'''

cv_cover = cv2.imread('../data/cv_cover.jpg')
for i in [5, 25, 35]:
	#Rotate Image
	cv_rotated = scipy.ndimage.rotate(cv_cover, i*10)
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(cv_cover, cv_rotated, opts)
	#Update histogram
	plotMatches(cv_cover, cv_rotated, matches, locs1, locs2)
	# pass # comment out when code is ready