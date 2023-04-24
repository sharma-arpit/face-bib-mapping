import os
from transformers import DetrImageProcessor, DetrForObjectDetection
import cv2
import shutil
from helper import *
from PIL import Image
import zipfile

bd_configPath = 'models/bib_detector/RBNR2_custom-yolov4-tiny-detector.cfg'
bd_weightsPath = 'models/bib_detector/RBNR2_custom-yolov4-tiny-detector_best.weights'
bd_classes = ['bib']
bd = Detector(bd_configPath, bd_weightsPath, bd_classes)
# True bounding box color
true_color = [15, 252, 75]
# Pred Bib bounding box color
color = [252, 15, 192]

human_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
human_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Define the local directory to download the photos to
download_dir = "/Users/arpitsharma/Downloads"
extract_dir = "/Users/arpitsharma/Downloads/TBT_ERROR"

# Define the directory to organize the photos into
organize_dir = "/Users/arpitsharma/Downloads/organized_photos"
if not os.path.exists(organize_dir):
    os.makedirs(organize_dir)

# Extract the photos from the ZIP archive
# with zipfile.ZipFile(os.path.join(download_dir, "photos.zip"), "r") as zip_ref:
#     if not os.path.exists(extract_dir):
#         os.makedirs(extract_dir)
#     zip_ref.extractall(extract_dir)

# Loop over all photos in the download directory
error = 0
for i, filename in enumerate(os.listdir(extract_dir)):

    filenumber = i + 1
    human = 0
    full_image = False
    bib = 0

    try:
        try:
            # Load the photo and convert it to grayscale
            img_cv2 = cv2.imread(os.path.join(extract_dir, filename))
            img_pil = Image.open(os.path.join(extract_dir, filename))
        except Exception as err:
            error += 1
            print(f"[ERROR] {filenumber} {filename}", err)
            continue

        if filenumber % 100 == 0:
            print(filenumber - error, "images processed out of", filenumber)

        try:
            inputs = human_processor(images=img_pil, return_tensors="pt")
            outputs = human_model(**inputs)

            target_sizes = torch.tensor([img_pil.size[::-1]])
            results = human_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)

        except Exception as err:
            error += 1
            # print(f"[ERROR] {filenumber} {filename} couldn't detect any human:", err)
            continue

        if len(results) == 0:
            error += 1
            # print(f"[ERROR] {filenumber} {filename} detected nothing.")
            continue

        for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
            box = [int(i) for i in box.tolist()]

            if human_model.config.id2label[label.item()] == "person" and round(score.item(), 3) > 0.9:
                human += 1
                human_cropped_pil = img_pil.crop((box[0], box[1], box[2], box[3]))
                human_cropped_cv2 = img_cv2[box[1]:box[3], box[0]:box[2]]

                bib_detections = bd.detect(human_cropped_cv2, 0.25)

                if len(bib_detections) == 0:
                    full_image = True
                    bib_detections_probables = bd.detect(img_cv2, 0.25)
                    bib_detections = []

                    for bib_box in bib_detections_probables:
                        (x, y, w, h) = bib_box[1]

                        if x >= box[0] and y >= box[1] and x+w <= box[2] and y+h <= box[3]:
                            bib_detections.append(['bib', [x, y, w, h]])

                if len(bib_detections):

                    for bib_box in bib_detections:
                        (x, y, w, h) = bib_box[1]

                        if full_image:
                            cropped_bib_pil = img_pil.crop((x, y, x + w, y + h))
                            cropped_bib_cv2 = img_cv2[y:y + h, x:x + w]
                        else:
                            cropped_bib_pil = human_cropped_pil.crop((x, y, x + w, y + h))
                            cropped_bib_cv2 = human_cropped_cv2[y:y + h, x:x + w]

                        bib_number = process_image(cropped_bib_pil)
                        try:
                            bib_number = int(bib_number)
                        except Exception as err:
                            # print(f"[ERROR] {filenumber} {filename} no bib detected.")
                            continue

                        if 999 < bib_number < 10000:
                            bib += 1
                            # Create the folder for the bib number if it does not exist
                            bib_dir = os.path.join(organize_dir, str(bib_number))
                            if not os.path.exists(bib_dir):
                                os.makedirs(bib_dir)
                            # Copy the photo to the folder with the same name as the bib number
                            shutil.copy(os.path.join(extract_dir, filename), os.path.join(bib_dir, filename))
                        else:
                            # print(f"[ERROR] {filenumber} {filename} invalid bib number range:", bib_number)
                            continue
                else:
                    # print(f"[ERROR] {filenumber} {filename} no bib detected.")
                    continue

        if human == 0 or bib == 0:
            error += 1
            print(f"[ERROR] {filenumber} {filename} no human or bib detected. humans: {human} & bibs: {bib}")
            continue

    except Exception as err:
        error += 1
        print("[ERROR]", filenumber, filename, err)

