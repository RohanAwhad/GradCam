from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys
from CAM import get_heatmap
import time

model = load_model(sys.argv[1])

def heatmap(image, orig_image):
    predictions = model.predict(image)[0]
    print(predictions)
    total_cam = np.zeros((32, 32, 3), dtype='float32')
    pred_label = np.argmax(predictions)
    total_cam += get_heatmap(model, [predictions, image], tape, "block5_pool", pred_label)
    for i in range(10):
        if i == pred_label:
            continue
        total_cam -= get_heatmap(model, [predictions, image], tape, "block5_pool", i)

    total_cam /= np.amx(np.abs(total_cam))
    total_cam = np.clip(total_cam, 0, 1)
    total_cam *= 255
    total_cam = np.uint8(total_cam)
    total_cam = Image.fromarray(total_cam).resize(orig_image.size)
    total_cam.save("%d.heatmap.png"%pred_label)

img = np.expand_dims(image.img_to_array(image.load_img(sys.argv[2], target_size=(32, 32, 3))), axis=0) / 255.0
heatmap(img, Image.open(sys.argv[2]))
