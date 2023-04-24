import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

text_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-stage1')
text_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-stage1')


class Detector:
    """
    Create YOLO object detection model in OpenCV with a given config and weights.
    Use this model to make predictions.

    Attributes
        classes (list): list of class names
        net (obj): openCV network object
        ln (obj): openCV layer names object
    """

    def __init__(self, cfg, wts, classes):
        """Initialize detector object

        Args
            cfg (str): path to model config file
            wts (str): path to model weights file
            classes (list): list of class names
        """

        self.classes = classes
        self.net = cv2.dnn.readNetFromDarknet(cfg, wts)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        # determine the output layer
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, img, conf):
        """
        Make predictions and return classes and bounding boxes

        Args
            img (numpy array): image array from opencv2 .imread
            conf (float): prediction confidence threshold

        Returns
            List containing bounding box values and class names for detections
            in the form [<class name>, [x, y, width, height]]
        """

        # format image for detection
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # get detections
        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)

        # initialize lists
        boxes = []
        confidences = []
        classIDs = []

        # initialize image dimensions
        h_img, w_img = img.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # drop low confidence detections and
                if confidence > conf:
                    box = detection[:4] * np.array([w_img, h_img, w_img, h_img])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non maximal suppression for
        # initialize lists
        self.boxes = []
        self.confidences = []
        self.detected_classes = []
        cls_and_box = []
        # get indices of final bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                self.boxes.append(boxes[i])
                self.confidences.append(confidences[i])
                self.detected_classes.append(self.classes[classIDs[i]])

                cls_and_box.append([self.classes[classIDs[i]], boxes[i]])

        return cls_and_box


def process_image(image):
    pixel_values = text_processor(image, return_tensors="pt").pixel_values
    generated_ids = text_model.generate(pixel_values)
    generated_text = text_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text
