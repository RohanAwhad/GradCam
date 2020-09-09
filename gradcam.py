from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import cv2

def _get_grad_model(model, layername):
    grad_model = Model(
    inputs=[model.inputs],
    outputs=[model.get_layer(layername).output,
             model.output])
    return grad_model

def _get_gradients(image, grad_model, class_idx):
    with tf.GradientTape() as tape:
        inputs = tf.cast(image, tf.float32)
        (conv_outputs, predictions) = grad_model(inputs)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    return grads, conv_outputs

def _get_class_activation_maps(conv_outputs, grads):
    cast_conv_outputs = tf.cast(conv_outputs > 0, tf.float32)
    cast_grads = tf.cast(grads > 0, tf.float32)
    guided_grads = cast_conv_outputs * cast_grads * grads

    # First axis is batch and it is equal to 1 so removing it
    conv_outputs = conv_outputs[0]
    guided_grads = guided_grads[0]

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

    return cam

def compute_heatmap(image, model, layername, class_idx, eps=1e-8):
    grad_model = _get_grad_model(model, layername)

    grads, conv_outputs = _get_gradients(image, grad_model, class_idx)

    cam = _get_class_activation_maps(conv_outputs, grads)

    w, h = image.shape[2], image.shape[1]
    heatmap = cv2.resize(cam.numpy(), (w, h))

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + eps)
    heatmap = (heatmap * 255).astype(np.uint8)

    return heatmap

def overlay_heatmap(heatmap, image, alpha=0.5, 
        colormap=cv2.COLORMAP_VIRIDIS):

    heatmap = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(image, alpha, heatmap, 1-alpha, 0)

    return (heatmap, output)


