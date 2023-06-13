import cv2
import os
import shutil
import numpy as np
import dlib
import face_recognition
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from keras_facenet import FaceNet

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
        self.boxes = []
        self.confidences = []
        self.detected_classes = []
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


def convert_opencv_to_dlib(bbox_opencv):
    x, y, width, height = bbox_opencv

    left = x
    top = y
    right = x + width
    bottom = y + height

    # Ensure the coordinates are within image bounds
    left = max(left, 0)
    top = max(top, 0)
    right = right
    bottom = bottom

    # Create a dlib rectangle object
    bbox_dlib = (left, top, right, bottom)

    return bbox_dlib


def is_tbt(bib_number):
    if 999 < bib_number < 10000 and (str(bib_number).startswith("12") or str(bib_number).startswith("65") or
                                     str(bib_number).startswith("30") or str(bib_number).startswith("31") or
                                     str(bib_number).startswith("66")):
        return True

    else:
        return False


def save_photo(organize_dir, extract_dir, bib_number, filename):
    bib_dir = os.path.join(organize_dir, str(bib_number))
    if not os.path.exists(bib_dir):
        os.makedirs(bib_dir)
    # Copy the photo to the folder with the same name as the bib number
    shutil.copy(os.path.join(extract_dir, filename), os.path.join(bib_dir, filename))


def draw_rectangle(img, boxes, font_color = (0, 255, 0), font_scale = 5.0, font = cv2.FONT_HERSHEY_SIMPLEX,
                   thickness = 10):

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 10)
        cv2.putText(img, str(i + 1), (x1, y1), font, font_scale, font_color, thickness)

    return img


class Participant:
    """
    A class to hold all possible data for a particular human participant found in an image.
    """

    def __init__(self, bib_number=None):

        self.bib_number = bib_number
        self.list_of_images = list()
        self.meta_data = dict()
        self.face_embeddings = {}


    def add_new_sample(self, filename, runner=None):

        self.meta_data[filename] = {"body": runner.body_location, "face": runner.face_location, "bib": runner.bib_location}
        self.face_embeddings[filename] = runner.face_vectors
        self.list_of_images.append(filename)


class Runner:

    def __init__(self, filename=None, img=None):

        self.filename = filename
        self.image = img
        self.body_location = None
        self.face_location = list()
        self.face_crops = None
        self.bib_location = None
        self.bib_number = None
        self.face_vectors = None
        self.identified = False
        self.detector = FaceNet()

    def detect_face(self, img=None, threshold=0.95):

        save = False
        if img is None:
            save = True
            (x1, y1, x2, y2) = self.body_location
            img = self.image[y1:y2, x1:x2].copy()

        detections, face_crops = self.detector.crop(img, threshold=threshold)
        face_location = []

        for d in detections:
            face_location.append(convert_opencv_to_dlib((d['box'][0] + x1, d['box'][1] + y1, d['box'][2], d['box'][3])))

        if save:
            self.face_location = face_location
            self.face_crops = face_crops

        return face_location, face_crops

    def embeddings(self, imgs=None):

        save = False
        if imgs is None:
            save = True
            imgs = self.face_crops

        face_vectors = self.detector.embeddings(imgs)

        if save:
            self.face_vectors = face_vectors

        return face_vectors

    def similarity(self, embedding1, embedding2):

        return self.detector.compute_distance(embedding1, embedding2)


    def __repr__(self):
        return f"{self.filename}:{self.bib_number}:{self.body_location}:{self.face_location}:{self.bib_location}"