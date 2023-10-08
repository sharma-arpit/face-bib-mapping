import os
from transformers import DetrImageProcessor, DetrForObjectDetection
import cv2
import shutil
from helper import *
import zipfile
import csv
import torch
import pandas as pd
import numpy as np


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

participants = open("participants.csv", mode='w')
nonparticipants = open("non_participants.csv", mode='w')

fields = ["filename","body_x1", "body_y1", "body_x2", "body_y2", "face_x1", "face_y1", "face_x2", "face_y2",
          "bib_x1", "bib_y1", "bib_x2", "bib_y2", "bib_number"]

runnerwriter = csv.DictWriter(participants, fieldnames=fields)
runnerwriter.writeheader()

nonrunnerwriter = csv.DictWriter(nonparticipants, fieldnames=fields)
nonrunnerwriter.writeheader()

embeddings_identified = open("embeddings_identified.csv", mode='w')
identifiedwriter = csv.writer(embeddings_identified)

embeddings_unidentified = open("embeddings_unidentified.csv", mode='w')
unidentifiedwriter = csv.writer(embeddings_unidentified)

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
                    runner.embeddings(img)                                                                               # Calculate the embedding vector for the detected faces

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

                                if runner.face_vectors is not None:
                                    runner.save(runnerwriter)
                                    identifiedwriter.writerow(list(runner.face_vectors[0]))

                                save_photo(organize_dir, photos_dir, bib_number, filename)

                                csvwriter.writerow([filename, bib_number, None, "bib"])

                            else:
                                if runner.identified and not runner.bib_number:

                                    if runner.face_vectors is not None:
                                        runner.save(nonrunnerwriter)
                                        unidentifiedwriter.writerow(list(runner.face_vectors[0]))

                                print(f"[ERROR] {filenumber} {filename} invalid bib number range:", bib_number)

                    else:

                        if runner.identified and not runner.bib_number:

                            if runner.face_vectors is not None:
                                runner.save(nonrunnerwriter)
                                unidentifiedwriter.writerow(list(runner.face_vectors[0]))

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

participants.close()
nonparticipants.close()
embeddings_identified.close()
embeddings_unidentified.close()

df_runner = pd.read_csv("participants.csv", header=0)
df_non_runner = pd.read_csv("non_participants.csv", header=0)

emb_runner = pd.read_csv("embeddings_identified.csv", header=None)
emb_non_runner = pd.read_csv("embeddings_unidentified.csv", header=None)

start_index = df_runner.columns.get_loc('bib_number')
identified = pd.concat([df_runner.iloc[:, start_index], emb_runner], axis=1).reindex(df_runner.index)
identified = identified.loc[~(emb_runner==0).all(axis=1)]

mean_emb = np.array(identified.groupby(by='bib_number',  sort=False).mean())
emb = np.array(emb_non_runner)

row_norms = np.linalg.norm(mean_emb, axis=1, keepdims=True)
mean_emb = mean_emb / row_norms

row_norms = np.linalg.norm(emb, axis=1, keepdims=True)
emb = emb / row_norms

result = 1 - np.dot(mean_emb, emb.T)
idx = list(np.argmin(result, axis=0))

indexs = list(identified.groupby(by='bib_number', sort=False).mean().index)
distance = 0.4

for j, i in enumerate(idx):

    if result[i][j] < distance:

        filename = df_non_runner['filename'][j]
        closest_bib = indexs[i]

        print(f"{filename} - {closest_bib}")
        save_photo(organize_dir, photos_dir, closest_bib, filename)
        # csvwriter.writerow([filename, bib_number, None, "face"])
