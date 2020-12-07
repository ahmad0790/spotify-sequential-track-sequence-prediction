import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(model_name, train_losses, valid_losses, train_accuracies, valid_accuracies):

	plt.figure()
	plt.plot(np.arange(len(train_losses)), train_losses, label='Train Loss')
	plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(loc="best")
	plt.savefig('output/' + model_name + '_loss_curve.png')

	plt.figure()
	plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Train Accuracy')
	plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(loc="best")
	plt.savefig('output/' + model_name + '_accuracy_curve.png')
