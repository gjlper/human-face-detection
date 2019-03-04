from abc import ABC, abstractmethod
import numpy as np
from joblib import Parallel, delayed

class Weak_Classifier(ABC):
	#initialize a harr filter with the positive and negative rects
	#rects are in the form of [x1, y1, x2, y2] 0-index
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		self.id = id
		self.plus_rects = plus_rects
		self.minus_rects = minus_rects
		self.num_bins = num_bins
		self.activations = None

	#take in one integrated image and return the value after applying the image
	#integrated_image is a 2D np array
	#return value is the number BEFORE polarity is applied
	def apply_filter2image(self, integrated_image):
		pos = 0
		for rect in self.plus_rects:
			rect = [int(n) for n in rect]
			pos += integrated_image[rect[3], rect[2]]\
				 + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1])\
				 - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]])\
				 - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
		neg = 0
		for rect in self.minus_rects:
			rect = [int(n) for n in rect]
			neg += integrated_image[rect[3], rect[2]]\
				 + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1])\
				 - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]])\
				 - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
		return pos - neg
	
		
	#take in a list of integrated images and calculate values for each image
	#integrated images are passed in as a 3-D np-array
	#calculate activations for all images BEFORE polarity is applied
	#only need to be called once
	def apply_filter(self, integrated_images):
		values = []
		for idx in range(integrated_images.shape[0]):
			values.append(self.apply_filter2image(integrated_images[idx, ...]))
		if (self.id + 1) % 100 == 0:
			print('Weak Classifier No. %d has finished applying' % (self.id + 1))
		return values
	
	#using this function to compute the error of
	#applying this weak classifier to the dataset given current weights
	#return the error and potentially other identifier of this weak classifier
	#detailed implementation is up you and depends
	#your implementation of Boosting_Classifier.train()
	@abstractmethod
	def calc_error(self, weights, labels):
		pass
	
	@abstractmethod
	def predict_image(self, integrated_image):
		pass

class Ada_Weak_Classifier(Weak_Classifier):
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		super().__init__(id, plus_rects, minus_rects, num_bins)
		self.polarity = None
		self.threshold = None
		self.preds = None

	def calc_error(self, weights, labels):
		######################
		######## TODO ########
		######################
		activations = self.activations
		threshold_list = np.linspace(activations.min(), activations.max(), self.num_bins+1)
		best_threshold = 0.
		best_polarity = 0.
		minErr = np.inf
		best_preds = None
		for polarity in [-1, 1]:
			for threshold in threshold_list:
				preds = polarity * np.sign(activations-threshold)
				err = np.sum((preds != labels).astype(int) * weights)
				if err < minErr:
					minErr = err
					best_threshold = threshold
					best_polarity = polarity
					best_preds = preds
		
		self.polarity = best_polarity
		self.threshold = best_threshold
		self.preds = best_preds
		return minErr
		
	def predict_image(self, integrated_image):
		value = self.apply_filter2image(integrated_image)
		return self.polarity * np.sign(value - self.threshold)

class Real_Weak_Classifier(Weak_Classifier):
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		super().__init__(id, plus_rects, minus_rects, num_bins)
		self.thresholds = None #this is different from threshold in ada_weak_classifier, think about it
		self.bin_pqs = None
		self.train_assignment = None

	def calc_error(self, weights, labels):
		######################
		######## TODO ########
		######################
		activations = self.activations
		thresholds = np.linspace(activations.min(), activations.max(), self.num_bins+1)
		thresholds[0] = -np.inf
		thresholds[-1] = np.inf
		digitized = np.digitize(activations, thresholds) - 1
		bin_pqs = np.zeros((2, self.num_bins))
		for bin_idx in range(self.num_bins):
			bin_p = np.sum(weights[(digitized == bin_idx) & (labels == 1)])
			bin_q = np.sum(weights[(digitized == bin_idx) & (labels == -1)])
			bin_pqs[:, bin_idx] = [bin_p, bin_q]
		
		Z = 2 * np.sum(np.sqrt(bin_pqs[0] * bin_pqs[1]))
		epsilon = 1e-10
        #Set epsilon incase bin_pqs equal to 0
		h = 1/2 * np.log((bin_pqs[0] + epsilon) / (bin_pqs[1] + epsilon))

		self.bin_pqs = bin_pqs
		self.thresholds = thresholds
		self.train_assignment = digitized
		self.h = h
		self.preds = h[digitized]
		self.polarity = 1
		return Z
	
	def predict_image(self, integrated_image):
		value = self.apply_filter2image(integrated_image)
		bin_idx = np.sum(self.thresholds < value) - 1
		return self.h[bin_idx]

def main():
	plus_rects = [(1, 2, 3, 4)]
	minus_rects = [(4, 5, 6, 7)]
	num_bins = 50
	ada_hf = Ada_Weak_Classifier(plus_rects, minus_rects, num_bins)
	real_hf = Real_Weak_Classifier(plus_rects, minus_rects, num_bins)

if __name__ == '__main__':
	main()
