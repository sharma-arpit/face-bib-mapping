import cv2
import os
import shutil
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from keras_facenet import FaceNet


face_detector = FaceNet()

class BibDetector:
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
        self.text_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-stage1')
        self.text_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-stage1')

    def detect(self, img, conf, swapRB=True, offset=None):
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
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=swapRB, crop=False)

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

                    if offset is not None:
                        x = x + offset[0]
                        y = y + offset[1]

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

                cls_and_box.append(boxes[i])

        return cls_and_box

    def de_duplicate(self, detections):

        remove_idx = []

        for i, detectionA in enumerate(detections):
            for j, detectionB in enumerate(detections):

                if i != j:
                    overlap, areaA, areaB = calculate_iou(convert_opencv_to_dlib(detectionA), convert_opencv_to_dlib(detectionB))
                    if overlap > 0.5:
                        if areaA > areaB:
                            remove_idx.append(j)
                        else:
                            remove_idx.append(i)

        return [item for i, item in enumerate(detections) if i not in list(set(remove_idx))]

    def process_image(self, image):
        """
        Extracts the text from the cropped bib image

        :param image (cv2.imread obj): cropped image of bib
        :return: text extracted from the image
        """

        pixel_values = self.text_processor(image, return_tensors="pt").pixel_values
        generated_ids = self.text_model.generate(pixel_values)
        generated_text = self.text_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text

class Participant:
    """
    A class to hold all available data for a particular human participant found in all images.
    """

    def __init__(self, bib_number=None):

        self.bib_number = bib_number
        self.list_of_images = list()
        self.meta_data = dict()
        self.face_embeddings = {}
        self.mean_embeddings = None


    def add_new_sample(self, filename, runner=None):
        """
        Adds a new sample found in the images for the same person based on the bib number
        :param filename (str): image file name
        :param runner (Runner object): a Runner object for a person from the image
        :return: None
        """

        self.meta_data[filename] = {"body": runner.body_location, "face": runner.face_location, "bib": runner.bib_location}
        if runner.face_vectors is not None:
            self.face_embeddings[filename] = runner.face_vectors[0]
        self.list_of_images.append(filename)

        return None

    def calculate_centroid(self):

        if self.face_embeddings.values():
            self.mean_embeddings = np.mean(np.array(list(self.face_embeddings.values())), axis=0)

        return self.mean_embeddings


class Runner:
    """
    A class to extract data for a particular runner in a particular image.
    """

    def __init__(self, filename=None, img=None):

        self.filename = filename
        self.body_location = None
        self.face_location = list()
        self.bib_location = None
        self.bib_number = None
        self.face_vectors = None
        self.identified = False

    def detect_face(self, img=None, threshold=0.95):
        """
        Detect faces in an image using MTCNN model based face detector
        :param img: cv2.imread object to detect faces
        :param threshold: confidence score for face detection
        :return: list of tuples containing face boxes and a list of cropped face arrays from the image
        """

        (x1, y1, x2, y2) = self.body_location

        detections, face_crops = face_detector.crop(img[y1:y2, x1:x2].copy(), threshold=threshold)
        face_location = list()

        for d in detections:
            face_location.append(convert_opencv_to_dlib((d['box'][0] + x1, d['box'][1] + y1, d['box'][2], d['box'][3])))

        if len(face_location):
            self.identified = True
            self.face_location = face_location

        return face_location, face_crops

    def embeddings(self, img=None, threshold=0.95):
        """
        Calculates the embeddings using FaceNet face descriptor
        :param imgs: list of cropped face arrays
        :return: a list of embeddings, one for each face
        """

        if img is not None:
            face_locations, imgs = self.detect_face(img, threshold=threshold)

            if len(face_locations):
                self.face_vectors = face_detector.embeddings(imgs)

        else:
            print("Image invalid.")

        return self.face_vectors

    def distance(self, embedding1, embedding2):
        """
        Calculate cosine distance between 2 face embedding vectors. Small distance for same person
        and large distance for different person
        :param embedding1: face vector for first face to compare
        :param embedding2: face vector for second face to compare
        :return: a float number which represent distance
        """

        return self.detector.compute_distance(embedding1, embedding2)

    def save(self, csvwriter):

        data = {'filename': self.filename,
                'body_x1': self.body_location[0],
                'body_y1': self.body_location[1],
                'body_x2': self.body_location[2],
                'body_y2': self.body_location[3],
                'bib_number': self.bib_number}

        if len(self.face_location) == 0:
            self.face_location.append((0, 0, 0, 0))

        data['face_x1'] = self.face_location[0][0]
        data['face_y1'] = self.face_location[0][1]
        data['face_x2'] = self.face_location[0][2]
        data['face_y2'] = self.face_location[0][3]

        if self.bib_location is None:
            self.bib_location = (0, 0, 0, 0)

        data['bib_x1'] = self.bib_location[0]
        data['bib_y1'] = self.bib_location[1]
        data['bib_x2'] = self.bib_location[2]
        data['bib_y2'] = self.bib_location[3]

        csvwriter.writerow(data)

    def __repr__(self):
        return f"{self.filename}::{self.bib_number}"


def convert_opencv_to_dlib(bbox_opencv):
    """
    Converts opencv format of bounding boxes (x, y, w, h) to dlib format bounding boxes (x1, y1, x2, y2)

    :param bbox_opencv (tuple): (x, y, w, h) bounding box
    :return: (tuple) (x1, y1, x2, y2) dlib format bounding box
    """
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


def is_correct(bib_number, event="TBT"):

    try:
        bib_number = int(bib_number)
    except ValueError as err:
        return False

    if event=="TBT":

        if 999 < bib_number < 10000 and (str(bib_number).startswith("12") or str(bib_number).startswith("65") or
                                         str(bib_number).startswith("30") or str(bib_number).startswith("31") or
                                         str(bib_number).startswith("66")):
            return True

        else:
            return False
    elif event=="SSU":

        if 999 < bib_number < 10000 and (str(bib_number).startswith("14") or str(bib_number).startswith("60") or
                                         str(bib_number).startswith("30") or str(bib_number).startswith("31") or
                                         str(bib_number).startswith("61") or str(bib_number).startswith("10")):
            return True

        else:
            return False

    return True


def save_photo(organize_dir, extract_dir, bib_number, filename):
    """
    Copy the photo in bib_number folder in organize_dir

    :param organize_dir (str): path to directory where file will be copied
    :param extract_dir (str): path to directory where original file is present
    :param bib_number (str): bib number (folder name) to copy the file
    :param filename (str): name of the file
    :return: None
    """

    bib_dir = os.path.join(organize_dir, str(bib_number))
    if not os.path.exists(bib_dir):
        os.makedirs(bib_dir)
    # Copy the photo to the folder with the same name as the bib number
    shutil.copy(os.path.join(extract_dir, filename), os.path.join(bib_dir, filename))

    return None

def draw_rectangle(img, boxes, font_color = (0, 255, 0), font_scale = 5.0, font = cv2.FONT_HERSHEY_SIMPLEX,
                   thickness = 10):
    """
    Draws boxes in the given image

    :param img (cv2.imread obj): image to draw boxes
    :param boxes (list): list of tuples containing locations of boxes
    :param font_color (tuple): color coding for the color of the boxes (B, G, R)
    :param font_scale (float): size of text to print in the image
    :param font (cv2.FONT_FAMILY obj): font family for the text
    :param thickness (int): thickness of the line in terms of number of pixel
    :return: a image with drawn boxes at given locations
    """

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 10)
        cv2.putText(img, str(i + 1), (x1, y1), font, font_scale, font_color, thickness)

    return img

def calculate_iou(boxA, boxB):
    """
    Calculate the intersection of union of bounding box A and bounding box B
    :param boxA: first bounding box
    :param boxB: second bounding box
    :return: percentage of intersection area, area A, area B
    """

    # Convert box coordinates to (x1, y1, x2, y2) format
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    # Calculate intersection area
    xA = max(x1A, x1B)
    yA = max(y1A, y1B)
    xB = min(x2A, x2B)
    yB = min(y2A, y2B)
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calculate union area
    boxA_area = (x2A - x1A + 1) * (y2A - y1A + 1)
    boxB_area = (x2B - x1B + 1) * (y2B - y1B + 1)
    union_area = boxA_area + boxB_area - intersection_area

    # Calculate IoU
    iou = intersection_area / float(union_area)

    return iou, boxA_area, boxB_area