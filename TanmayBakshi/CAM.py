from tensorflow import keras
import numpy as np

def get_heatmap(model, pred_img, tape, last_layer, heatmap_for_index=None):
  last_conv_layer = model.layers[0].get_layer(last_layer)
  prediction = pred_img[0]
  x = pred_img[1]
  class_idx = np.argmax(prediction)
  if heatmap_for_index is None:
    heatmap_for_index = class_idx
  class_output = model.output[:, heatmap_for_index]
  grads = keras.backend.gradients(class_output, last_conv_layer.output)[0]
  pooled_grads = keras.backend.mean(grads, axis=(0, 1, 2))
  iterate = keras.backend.function([model.input], [pooled_grads, last_conv_layer.output[0]])
  for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
  heatmap = np.mean(conv_layer_output_value, axis=-1)
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)
  heatmap = keras.backend.resize_images(heatmap, (32, 32))
  heatmap = np.uint8(255 * heatmap)
  heatmap = np.repeat(heatmap[:, :, np.newaxis], 3, axis=2)
  return heatmap
