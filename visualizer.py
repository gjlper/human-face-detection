import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from PIL import Image, ImageDraw

class Visualizer:
	def __init__(self, histogram_intervals, top_wc_intervals):
		self.histogram_intervals = histogram_intervals
		self.top_wc_intervals = top_wc_intervals
		self.weak_classifier_accuracies = {}
		self.strong_classifier_scores = {}
		self.top_wc_errors = {}
		self.filters = []
		self.sc_errors = {}
		self.labels = None
	
	def draw_sc_errors(self):
		plt.figure()
		plt.plot(self.sc_errors.keys(), self.sc_errors.values())
		plt.title('Training Error of Strong Classifier')
		plt.savefig('Training Error of Strong Classifier')

	def draw_histograms(self):
		for t in self.strong_classifier_scores:
			scores = self.strong_classifier_scores[t]
			pos_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == 1]
			neg_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == -1]

			bins = np.linspace(np.min(scores), np.max(scores), 100)

			plt.figure()
			plt.hist(pos_scores, bins, alpha=0.5, label='Faces',color = 'palevioletred')
			plt.hist(neg_scores, bins, alpha=0.5, label='Non-Faces',color = 'seagreen')
			plt.legend(loc='upper right')
			plt.title('Using %d Weak Classifiers' % t)
			plt.savefig('histogram_%d.png' % t)

	def draw_rocs(self):
		plt.figure()
		for t in self.strong_classifier_scores:
			scores = self.strong_classifier_scores[t]
			fpr, tpr, _ = roc_curve(self.labels, scores)
			plt.plot(fpr, tpr, label = 'No. %d Weak Classifiers' % t)
		plt.legend(loc = 'lower right')
		plt.title('ROC Curve')
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.savefig('ROC Curve')

	def draw_wc_errors(self):
		plt.figure()
		for t in self.top_wc_errors:
			errors = self.top_wc_errors[t]
			plt.plot(errors, label = 'Step T= %d ' % t)
		plt.ylabel('Error')
		plt.xlabel('Weak Classifiers')
		plt.title('Top 1000 Weak Classifier Errors')
		plt.legend(loc = 'lower right')
		plt.savefig('Weak Classifier Errors')

	def draw_haar_filters(self):
		plt.figure(figsize=(20, 4))
		n_filters = 20
		for i in range(n_filters):
			plt.subplot(2, n_filters/2, i+1)
			haar_filter = self.filters[i]
			im = Image.new('L', (16, 16), 128) 
			draw = ImageDraw.Draw(im)
			for plus_rec in haar_filter[0]:
				draw.rectangle(plus_rec, fill=255, outline=255)
			for minus_rec in haar_filter[1]:
				draw.rectangle(minus_rec, fill=0, outline=0)
			plt.imshow(im)
			alpha = haar_filter[2]
			plt.title('{:.2f}'.format(alpha))
			plt.axis('off')
		plt.savefig('Top %d Haar Filters'%n_filters, bbox_inches = 'tight', pad_inches = 0)



	def draw_wc_accuracies(self):
		plt.figure()
		for t in self.weak_classifier_accuracies:
			accuracies = self.weak_classifier_accuracies[t]
			plt.plot(accuracies, label = 'After %d Selection' % t)
		plt.ylabel('Accuracy')
		plt.xlabel('Weak Classifiers')
		plt.title('Top 1000 Weak Classifier Accuracies')
		plt.legend(loc = 'upper right')
		plt.savefig('Weak Classifier Accuracies')

if __name__ == '__main__':
	main()
