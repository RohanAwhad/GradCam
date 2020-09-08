from gradcam import GradCam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="path to input image")

args = vars(ap.parse_args())

model = VGG16()
orig = cv2.cvtColor(cv2.imread(args["image"]), cv2.COLOR_BGR2RGB)
orig = cv2.resize(orig, (224, 224))
image = np.expand_dims(orig, axis=0)
image = preprocess_input(image)

preds = model.predict(image)
target_class = np.argmax(preds[0])
print(decode_predictions(preds, top=3))
print(f'Target Class: {target_class}')

cam = GradCam(model, target_class, 'block5_conv')
heatmap = cam.compute_heatmap(image)

(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

output = np.vstack([orig, heatmap, output])
output = cv2.resize(output, (224, 700))
cv2.imshow("output", output)
cv2.waitKey(0)
