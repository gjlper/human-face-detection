import numpy as np
import time
import cv2
from boosting_classifier import Boosting_Classifier
from visualizer import Visualizer
from im_process import normalize
from utils import *

def main():
	#flag for debugging
	flag_subset = False
	boosting_type = 'Real' #'Real' or 'Ada'
	training_epochs = 100 if not flag_subset else 20
	act_cache_dir = 'wc_activations.npy' if not flag_subset else 'wc_activations_subset.npy'
	chosen_wc_cache_dir = 'chosen_wcs.pkl' if not flag_subset else 'chosen_wcs_subset.pkl'

	#data configurations
	pos_data_dir = 'newface16'
	neg_data_dir = 'nonface16'
	image_w = 16
	image_h = 16
	data, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, flag_subset)
	data = integrate_images(normalize(data))

	#number of bins for boosting
	num_bins = 25

	#number of cpus for parallel computing
	num_cores = 6 if not flag_subset else 1 #always use 1 when debugging
	
	#create Haar filters
	filters = generate_Haar_filters(4, 4, 16, 16, image_w, image_h, flag_subset)

	#create visualizer to draw histograms, roc curves and best weak classifier accuracies
	drawer = Visualizer([10, 20, 50, 100], [1, 10, 20, 50, 100])
	
	#create boost classifier with a pool of weak classifier
	boost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)

	# #calculate filter values for all training images
     start = time.clock()
     boost.calculate_training_activations(act_cache_dir, act_cache_dir)
     end = time.clock()
     print('%f seconds for activation calculation' % (end - start))

     print('Training the boosting classifier...')
     if boosting_type == 'Real':
         boost.train_real_boosting(chosen_wc_cache_dir)
     else:
         boost.train(chosen_wc_cache_dir)

     do_hard_negative_mining = False
     if do_hard_negative_mining:
         print('do hard negative mining ...')
         boost.load_trained_wcs('chosen_wcs.pkl')
         background_img = cv2.imread('./Testing_Images/Non_face_1.jpg', cv2.IMREAD_GRAYSCALE)
         scale_step = 20 if not flag_subset else 1
         hard_negative_patches = boost.get_hard_negative_patches(background_img, scale_step)
         print(boost.data.shape, hard_negative_patches.shape)
         boost.data = np.concatenate([boost.data, hard_negative_patches], axis=0)
         neg_labels = -1 * np.ones(hard_negative_patches.shape[0])
         boost.labels = np.concatenate([boost.labels, neg_labels])
         act_cache_dir = 'wc_activations_hard-negative-mining.npy'
         chosen_wc_cache_dir = 'chosen_wcs_hard-negative-mining.pkl'
         boost.calculate_training_activations(act_cache_dir, act_cache_dir)
         boost.train(chosen_wc_cache_dir)

     boost.visualize()

     #boost.load_trained_wcs('chosen_wcs_hard-negative-mining.pkl')

	for img in ['Face_1', 'Face_2', 'Face_3']:
		print(img)
		original_img = cv2.imread('./Testing_Images_2018/%s.jpg'%img, cv2.IMREAD_GRAYSCALE)
		scale_step = 20 if not flag_subset else 1
		result_img = boost.face_detection(original_img, scale_step)
		cv2.imwrite('Result_img_%s_%s.png' % (boosting_type, img), result_img)

if __name__ == '__main__':
	main()
