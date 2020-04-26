import numpy as np

# def accuracy(images, targets, params):
#   target_class = np.argmax(targets, axis=1)
#   predicted_class = np.argmax(batched_forward(images, params), axis=1)
#   return np.mean(predicted_class == target_class)

#compute the accuracy
def accuracy(preds, targets):
  target_class = np.argmax(targets, axis=1)
  predicted_class = np.argmax(preds, axis=1)
  return np.mean(predicted_class == target_class)
