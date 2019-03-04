import os
import numpy as np
from tqdm import tqdm, trange
from joblib import Parallel, delayed
import pickle
from copy import deepcopy

import cv2
from weak_classifier import Ada_Weak_Classifier, Real_Weak_Classifier
from im_process import image2patches, nms, normalize

class Boosting_Classifier:
	def __init__(self, haar_filters, data, labels, num_chosen_wc, num_bins, visualizer, num_cores, style):
		self.filters = haar_filters
		self.data = data
		self.labels = labels
		self.num_chosen_wc = num_chosen_wc
		self.num_bins = num_bins
		self.visualizer = visualizer
		self.num_cores = num_cores
		self.style = style
		self.chosen_wcs = []
		if style == 'Ada':
			self.weak_classifiers = [Ada_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
		elif style == 'Real':
			self.weak_classifiers = [Real_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
	
	def calculate_training_activations(self, save_dir = None, load_dir = None):
		print('Calcuate activations for %d weak classifiers, using %d images.' % (len(self.weak_classifiers), self.data.shape[0]))
		if load_dir is not None and os.path.exists(load_dir):
			print('[Find cached activations, %s loading...]' % load_dir)
			wc_activations = np.load(load_dir)
		else:
			if self.num_cores == 1:
				wc_activations = [wc.apply_filter(self.data) for wc in self.weak_classifiers]
			else:
				wc_activations = Parallel(n_jobs = self.num_cores)(delayed(wc.apply_filter)(self.data) for wc in self.weak_classifiers)
			wc_activations = np.array(wc_activations)
			if save_dir is not None:
				print('Writing results to disk...')
				np.save(save_dir, wc_activations)
				print('[Saved calculated activations to %s]' % save_dir)
		for wc in self.weak_classifiers:
			wc.activations = wc_activations[wc.id, :]
		return wc_activations
	
	#select weak classifiers to form a strong classifier
	#after training, by calling self.sc_function(), a prediction can be made
	#self.chosen_wcs should be assigned a value after self.train() finishes
	#call Weak_Classifier.calc_error() in this function
	#cache training results to self.visualizer for visualization
	#
	#
	#detailed implementation is up to you
	#consider caching partial results and using parallel computing
	def train(self, save_dir = None):
		######################
		######## TODO ########
		######################
		labels = self.labels
		self.chosen_wcs = []

		vis = self.visualizer
		top_wc_intervals = vis.top_wc_intervals + [-1]
		histogram_intervals = vis.histogram_intervals + [-1]
		t_log_top_wc = top_wc_intervals.pop(0)
		t_log_sc_scores = histogram_intervals.pop(0)

		D = np.ones(self.data.shape[0]) * 1.0 / self.data.shape[0]
		for t in trange(self.num_chosen_wc):
			if self.num_cores == 1:
				res = [wc.calc_error(D, labels) for wc in self.weak_classifiers]
			else:
				res = Parallel(n_jobs=self.num_cores)(delayed(wc.calc_error)(D, labels) for wc in self.weak_classifiers)
			error_list = res
			
			best_wc_id = np.argmin(error_list)
			best_error = error_list[best_wc_id]
			best_wc = self.weak_classifiers[best_wc_id]
			best_wc.calc_error(D, labels)

			if self.style == 'Ada':
				alpha = 0.5 * np.log((1.0-best_error)/best_error)
			elif self.style == 'Real':
				alpha = 1.0
			self.chosen_wcs.append((alpha, deepcopy(best_wc)))
			best_preds = best_wc.preds
			D *= np.exp(-alpha * best_preds * labels)
			D /= D.sum()


			####### log #######
			if t+1 == t_log_top_wc:
				vis.top_wc_errors[t+1] = sorted(error_list)[:1000]
				t_log_top_wc = top_wc_intervals.pop(0)
			
			if t % 1 == 0:
				vis.sc_errors[t+1] = self.sc_error()[0]

			if t+1 == t_log_sc_scores:
				vis.strong_classifier_scores[t+1] = self.sc_error()[1]
				t_log_sc_scores = histogram_intervals.pop(0)
			######## log ######

		# vis.filters = [(wc.plus_rects, wc.minus_rects, alpha) for alpha, wc in self.chosen_wcs]
		vis.filters = []
		for alpha, wc in self.chosen_wcs:
			if wc.polarity == 1:
				vis.filters.append((wc.plus_rects, wc.minus_rects, alpha))
			elif wc.polarity == -1:
				vis.filters.append((wc.minus_rects, wc.plus_rects, alpha))


		if save_dir is not None:
			pickle.dump(self.chosen_wcs, open(save_dir, 'wb'))

	# using the filters chosen in AdaBoosting
	def train_real_boosting(self, save_dir=None):
		labels = self.labels
		self.chosen_wcs = []
		chosen_wcs_ada = pickle.load(open('chosen_wcs.pkl', 'rb'))

		vis = self.visualizer
		top_wc_intervals = vis.top_wc_intervals + [-1]
		histogram_intervals = vis.histogram_intervals + [-1]
		t_log_top_wc = top_wc_intervals.pop(0)
		t_log_sc_scores = histogram_intervals.pop(0)

		D = np.ones(self.data.shape[0]) * 1.0 / self.data.shape[0]
		for t in trange(self.num_chosen_wc):
			best_wc_id = chosen_wcs_ada[t][1].id
			best_wc = self.weak_classifiers[best_wc_id]
			best_error = best_wc.calc_error(D, labels)
			alpha = 1.0
			self.chosen_wcs.append((alpha, deepcopy(best_wc)))
			best_preds = best_wc.preds
			D *= np.exp(-alpha * best_preds * labels)
			D /= D.sum()

			####### log #######
			if t+1 == t_log_top_wc:
				vis.top_wc_errors[t+1] = [best_error] * 1000
				t_log_top_wc = top_wc_intervals.pop(0)
			
			if t % 1 == 0:
				vis.sc_errors[t+1] = self.sc_error()[0]

			if t+1 == t_log_sc_scores:
				vis.strong_classifier_scores[t+1] = self.sc_error()[1]
				t_log_sc_scores = histogram_intervals.pop(0)
			######## log ######


		# vis.filters = [(wc.plus_rects, wc.minus_rects) for _, wc in self.chosen_wcs]
		vis.filters = []
		for alpha, wc in self.chosen_wcs:
			if wc.polarity == 1:
				vis.filters.append((wc.plus_rects, wc.minus_rects, alpha))
			elif wc.polarity == -1:
				vis.filters.append((wc.minus_rects, wc.plus_rects, alpha))

		if save_dir is not None:
			pickle.dump(self.chosen_wcs, open(save_dir, 'wb'))

	def sc_error(self):
		scores = 0
		for alpha, wc in self.chosen_wcs:
			scores += alpha * wc.preds
		preds = np.sign(scores)
		error = np.mean(preds != self.labels)
		return error, scores

	def sc_function(self, image):
		return np.sum([np.array([alpha * wc.predict_image(image) for alpha, wc in self.chosen_wcs])])			

	def load_trained_wcs(self, save_dir):
		self.chosen_wcs = pickle.load(open(save_dir, 'rb'))	

	def face_detection(self, img, scale_step = 20):
		
		# this training accuracy should be the same as your training process,
		##################################################################################
		train_predicts = []
		for idx in range(self.data.shape[0]):
			train_predicts.append(self.sc_function(self.data[idx, ...]))
		print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))
		##################################################################################

		scales = 1 / np.linspace(1, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)
		print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
		predicts = [self.sc_function(patch) for patch in tqdm(patches)]
		print(np.mean(np.array(predicts) > 0), np.sum(np.array(predicts) > 0))
		pos_predicts_xyxy = np.array([patch_xyxy[idx] + [score] for idx, score in enumerate(predicts) if score > 0])
		if pos_predicts_xyxy.shape[0] == 0:
			return
		xyxy_after_nms = nms(pos_predicts_xyxy, 0.01)
		
		print('after nms:', xyxy_after_nms.shape[0])
		for idx in range(xyxy_after_nms.shape[0]):
			pred = xyxy_after_nms[idx, :]
			cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 2) #green rectangular with line width 3

		return img

	def get_hard_negative_patches(self, img, scale_step = 10):
		scales = 1 / np.linspace(1, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)
		print('Get Hard Negative in Progress ..., total %d patches' % patches.shape[0])
		predicts = np.array([self.sc_function(patch) for patch in tqdm(patches)])
		wrong_patches = patches[np.where(predicts > 0)[0], ...]
		# wrong_patches = patches[:100]

		return wrong_patches

	def visualize(self):
		self.visualizer.labels = self.labels
		self.visualizer.draw_histograms()
		self.visualizer.draw_rocs()
		# self.visualizer.draw_wc_accuracies()
		self.visualizer.draw_wc_errors()
		self.visualizer.draw_haar_filters()
		self.visualizer.draw_sc_errors()
