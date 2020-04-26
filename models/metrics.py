import numpy as np

#compute the accuracy 
def accuracy(preds, targets):
  target_class = np.argmax(targets, axis=1)
  predicted_class = np.argmax(preds, axis=1)
  return np.mean(predicted_class == target_class)
