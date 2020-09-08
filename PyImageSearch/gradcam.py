from tensorflow.keras.models import Model
import cv2
import numpy as np
import tensorflow as tf

class GradCam:
    def __init__(self, model, class_idx, layername=None):
        self.model = model
        self.class_idx = class_idx
        self.layername = layername

    def compute_heatmap(self, image, eps=1e-8):
        grad_model = Model(
        inputs=[self.model.inputs],
        outputs=[self.model.get_layer(self.layername).output,
                 self.model.output])

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (conv_outputs, predictions) = grad_model(inputs)
            loss = predictions[:, self.class_idx]

        grads = tape.gradient(loss, conv_outputs)

        cast_conv_outputs = tf.cast(conv_outputs > 0, tf.float32)
        cast_grads = tf.cast(grads > 0, tf.float32)
        guided_grads = cast_conv_outputs * cast_grads * grads

        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        w, h = image.shape[2], image.shape[1]
        heatmap = cv2.resize(cam.numpy(), (w, h))

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + eps)
        heatmap = (heatmap * 255).astype(np.uint8)

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, 
            colormap=cv2.COLORMAP_VIRIDIS):

        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1-alpha, 0)

        return (heatmap, output)


