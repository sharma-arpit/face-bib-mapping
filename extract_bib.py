import os
from transformers import DetrImageProcessor, DetrForObjectDetection
import cv2
import shutil
from helper import *
import zipfile
import csv
from scipy import spatial
import torch
import pandas as pd


bd_configPath = 'models/bib_detector/RBNR2_custom-yolov4-tiny-detector.cfg'
bd_weightsPath = 'models/bib_detector/RBNR2_custom-yolov4-tiny-detector_best.weights'
bd_classes = ['bib']
bd = BibDetector(bd_configPath, bd_weightsPath, bd_classes)

human_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
human_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Define the local directory to download the photos to
photos_dir = "/Users/arpitsharma/Downloads/TBT"

# Define the directory to organize the photos into
organize_dir = "/Users/arpitsharma/Downloads/organized_photos"
if not os.path.exists(organize_dir):
    os.makedirs(organize_dir)

# Loop over all photos in the download directory
error = 0
participants = {}
unidentified_participants = {}

with open("results.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["filename", "bib_number", "error", "mode"])

    for i, filename in enumerate(os.listdir(photos_dir)):

        filenumber = i + 1
        human = 0
        full_image = False
        bib = 0

        try:
            try:
                img = cv2.cvtColor(cv2.imread(os.path.join(photos_dir, filename)), cv2.COLOR_BGR2RGB)
            except Exception as err:
                error += 1
                csvwriter.writerow([filename, "", err])
                print(f"[ERROR] {filenumber} {filename}", err)
                continue

            if filenumber % 100 == 0:
                print(filenumber - error, "images processed out of", filenumber)

            try:
                inputs = human_processor(images=img, return_tensors="pt")
                outputs = human_model(**inputs)

                target_sizes = torch.tensor([img.shape[:2]])
                results = human_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)

            except Exception as err:
                error += 1
                print(f"[ERROR] {filenumber} {filename} couldn't detect any human:", err)
                continue

            if len(results) == 0:
                error += 1
                print(f"[ERROR] {filenumber} {filename} detected nothing.")
                continue

            bib_detections_probables = bd.detect(img, 0.20, swapRB=True) + bd.detect(img, 0.20, swapRB=False)
            bib_detections_probables = bd.de_duplicate(bib_detections_probables)
            print(filename, bib_detections_probables)

            for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):

                if human_model.config.id2label[label.item()] == "person" and round(score.item(), 3) > 0.9:
                    body_box = [int(i) for i in box.tolist()]
                    human += 1

                    runner = Runner(filename=filename)
                    runner.body_location = body_box                                                                      # Save the body box of the runner
                    runner.embeddings(img)                                                                                  # Calculate the embedding vector for the detected faces

                    bib_detections = []

                    for bib_box in bib_detections_probables:
                        (x, y, w, h) = bib_box

                        if x >= body_box[0] and y >= body_box[1] and x + w <= body_box[2] and y + h <= body_box[3]:
                            bib_detections.append((x, y, w, h))

                    if len(bib_detections) == 0:
                        body = img[body_box[1]:body_box[3], body_box[0]:body_box[2]]
                        bib_detections = bd.detect(body, 0.2, swapRB=True, offset=body_box) + bd.detect(body, 0.2,
                                                                                                        swapRB=False,
                                                                                                        offset=body_box)
                        bib_detections = bd.de_duplicate(bib_detections)

                        print(filename, bib_detections)

                    if len(bib_detections):

                        for bib_box in bib_detections:
                            (x1, y1, x2, y2) = convert_opencv_to_dlib(bib_box)
                            runner.bib_location = (x1, y1, x2, y2)

                            bib_number = bd.process_image(img[y1:y2, x1:x2])

                            if is_correct(bib_number, event="TBT"):
                                bib += 1
                                runner.bib_number = bib_number
                                runner.identified = True
                                print("Found bib number:", bib_number)

                                if str(bib_number) not in participants.keys():
                                    participants[str(bib_number)] = Participant(bib_number=bib_number)

                                participants[str(bib_number)].add_new_sample(filename, runner=runner)
                                runner = None
                                save_photo(organize_dir, photos_dir, bib_number, filename)

                                csvwriter.writerow([filename, bib_number, None, "bib"])

                            else:
                                if filename not in unidentified_participants.keys() and runner.identified:
                                    unidentified_participants[filename] = list()

                                if runner.identified:
                                    unidentified_participants[filename].append(runner)
                                else:
                                    runner = None

                                print(f"[ERROR] {filenumber} {filename} invalid bib number range:", bib_number)

                    else:
                        if filename not in unidentified_participants.keys() and runner.identified:
                            unidentified_participants[filename] = list()

                        if runner.identified:
                            unidentified_participants[filename].append(runner)
                        else:
                            runner = None

                        print(f"[ERROR] {filenumber} {filename} no bib detected.")
                        continue

            if human == 0 or bib == 0:
                error += 1
                err = f"[ERROR] {filenumber} {filename} no human or bib detected. humans: {human} & bibs: {bib}"
                csvwriter.writerow([filename, "", err])
                print(err)

        except Exception as err:
            error += 1
            csvwriter.writerow([filename, "", err])
            print("[ERROR]", filenumber, filename, err)

    print("Face Clustering Started.....")

    # Use Face features to match unidentified images
    for participant in participants:
        participant.calculate_centroid()

    for filename in unidentified_participants.keys():

        for runner in unidentified_participants[filename]:

            average_distances = {}
            closest = 100
            closest_bib = None

            for bib in participants.keys():

                average_distances[bib] = spatial.distance.cosine(participants[bib].mean_embeddings, runner.face_vectors[0])

                if average_distances[bib] < closest:
                    closest_bib = bib
                    closest = average_distances[bib]

            if closest < 0.4:
                print(f"{filename} - {closest_bib}")
                save_photo(organize_dir, photos_dir, closest_bib, filename)
                csvwriter.writerow([filename, bib_number, None, "face"])