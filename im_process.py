from sklearn.feature_extraction import image as IMG
import numpy as np
import cv2
from utils import integrate_images

#extract patches from the image for all scales of the image
#return the INTEGREATED images and the coordinates of the patches
def image2patches(scales, image, patch_w = 16, patch_h = 16):
	all_patches = np.zeros((0, patch_h, patch_w))
	all_x1y1x2y2 = []
	for s in scales:
		simage = cv2.resize(image, None, fx = s, fy = s, interpolation = cv2.INTER_CUBIC)
		height, width = simage.shape
		print('Image shape is: %d X %d' % (height, width))
		patches = IMG.extract_patches_2d(simage, (patch_w, patch_h)) # move along the row first

		total_patch = patches.shape[0]
		row_patch = (height - patch_h + 1)
		col_patch = (width - patch_w + 1)
		assert(total_patch == row_patch * col_patch)
		scale_xyxy = []
		for pid in range(total_patch):
			y1 = pid / col_patch
			x1 = pid % col_patch
			y2 = y1 + patch_h - 1
			x2 = x1 + patch_w - 1
			scale_xyxy.append([int(x1 / s), int(y1 / s), int(x2 / s), int(y2 / s)])
		all_patches = np.concatenate((all_patches, patches), axis = 0)
		all_x1y1x2y2 += scale_xyxy
	return integrate_images(normalize(all_patches)), all_x1y1x2y2

#return a vector of prediction (0/1) after nms, same length as scores
#input: [x1, y1, x2, y2, score], threshold used for nms
#output: [x1, y1, x2, y2, score] after nms
def nms(xyxys, overlap_thresh):
	######################
	######## TODO ########
	######################
	x1 = xyxys[:,0]
	y1 = xyxys[:,1]
	x2 = xyxys[:,2]
	y2 = xyxys[:,3]
	scores = xyxys[:,4]
    #compute the area of the bounding boxes and sort the bounding
    #boxes by scores
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = np.argsort(scores)[::-1]
	keep = []
	while order.size > 0:
		i = order[0]
		j = order[1:]
		keep.append(i)
        #find the largest (x,y) coordinates for the start of
        #the bounding box and the smallest (x,y) coordinates
        #for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[j])
		yy1 = np.maximum(y1[i], y1[j])
		xx2 = np.minimum(x2[i], x2[j])
		yy2 = np.minimum(y2[i], y2[j])
        #compute the width and height of the bounding box
		w = np.maximum(0., xx2 - xx1 + 1)
		h = np.maximum(0., yy2 - yy1 + 1)
        #compute the ratio of overlap
		inter = w * h
		ovr = inter / (areas[i] + areas[j] - inter)
        #delete all indexes from the index list
		inds = np.where(ovr <= overlap_thresh)[0]
		order = order[inds + 1]

	xyxys = xyxys[keep]
	return xyxys

def normalize(images):
	images = (images - np.min(images)) / (np.max(images) - np.min(images))
	return images

def main():
	original_img = cv2.imread('Testing_Images_2018/Face_1.jpg', cv2.IMREAD_GRAYSCALE)
	scales = 1 / np.linspace(1, 10, 46)
	patches, patch_xyxy = image2patches(scales, original_img)
	print(patches.shape)
	print(len(patch_xyxy))
if __name__ == '__main__':
	main()
