import cv2
import numpy as np

import onnx
import onnxruntime as ort
# import onnx_tf

from imutils.video import WebcamVideoStream

HEIGHT, WIDTH = 480, 640

def preprocess_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (WIDTH, HEIGHT))
    img_mean = np.array([127,127,127])
    img = (img - img_mean) / 128 # so that pixel = (0,1)
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis = 0)
    img = img.astype(np.float32)

    return img

### Compute the IOU of the picked boxes with the rest of the boxes
def iou_of(box_1, box_2):
    startX_IOU = 0
    startY_IOU = 0
    endX_IOU = 0
    endY_IOU = 0
    # print(box_1)

    (startX1, startY1, endX1, endY1) = box_1
    (startX2, startY2, endX2, endY2) = box_2
    w_1, h_1 = endX1 - startX1, endY1 - startY1
    w_2, h_2 = endX2 - startX2, endY2 - startY2
    
    if(startX1 > startX2):
        startX_IOU = startX1
        endX_IOU = endX2
    else:
        startX_IOU = startX2
        endX_IOU = endX1

    if(startY1 > startY2):
        startY_IOU = startY1
        endY_IOU = endY2
    else:
        startY_IOU = startY2
        endY_IOU = endY1

    w_iou = endX_IOU - startX_IOU
    h_iou = endY_IOU - startY_IOU

    iou = w_iou * h_iou
    union = (w_1 * h_1 + w_2 * h_2) - iou

    iou = iou / union

    return iou

def hard_nms(boxes, confidences, iou_threshold=0.2):
    # discarded = list()
    # boxes = boxes[0]
    # confidences = confidences[0]
    
    # print(confidences)
    boxes = boxes[np.where(confidences[:,1] > 0.8)]
    confidences = confidences[np.where(confidences[:,1] > 0.8)]

    picked = list()

    indexes = np.argsort(confidences)
    if(len(indexes) == 0):
        return picked

    picked.append(boxes[indexes[::-1][0]]) # pick the top scored box first

    while(True):
        discarded = list()
        for i in range(len(picked)):
            for other in boxes[indexes[::-1][1:]]:
                print(picked[i], other)
                iou = iou_of(picked[i], other)
                if(iou > 0.2):
                    discarded.append(other)
                else:
                    picked.append(other)

        if(len(discarded) == 0):
            break

    return picked

onnx_path = 'ultra_light/ultra_light_models/ultra_light_640.onnx'
onnx_model = onnx.load(onnx_path)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

vs = WebcamVideoStream(src=0).start()

while(True):
    frame = vs.read()
    H, W = frame.shape[:2]
    img = preprocess_img(frame)

    confidences, boxes = ort_session.run(None, {input_name:img})
    confidences, boxes = confidences[0], boxes[0]
    boxes[:, 0] *= W
    boxes[:, 1] *= H
    boxes[:, 2] *= W
    boxes[:, 3] *= H

    boxes = hard_nms(boxes, confidences)

    print(boxes)
