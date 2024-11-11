import numpy as np
from scipy.integrate import simps


def roc_auc(pred, target, n_points=20, include_prior=False):
        """
        Calculates the Reciever-Operating-Characteristic (ROC) area under
        the curve (AUC) by numerical integration.
        """

        target = np.array(target)/255        
        generated = pred
        # min max normalisation
        generated = (generated - generated.min())/(generated.max() - generated.min())

        def roc(p=0.1):
            x = generated.reshape(-1) > p
            t = target.reshape(-1) > p

            return np.sum(x==t)/len(t)

        calculate_roc = np.vectorize(roc)

        x = np.linspace(0, 1, n_points)
        auc = simps(calculate_roc(x))/n_points

        return auc

def calculate_auc(preds, targets):
	"""
	inputs -- 2 dictionary with prediction and target images. The 2 dictionaries have the  same number of keys, where each key identifies an unique image. 
	The predictions have the predicted fixation maps while the targets have the ground truth fixation maps which are available from "https://people.csail.mit.edu/tjudd/WherePeopleLook/" 
	"""
	assert preds.keys() == targets.keys()
	mean_auc = 0
	for key in preds.keys():
		mean_auc += roc_auc(preds[key], targets[key])
	mean_auc /= len(preds.keys())
	return mean_auc
