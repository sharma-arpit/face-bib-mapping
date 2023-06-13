import os
from transformers import DetrImageProcessor, DetrForObjectDetection
import cv2
import shutil
from helper import *
from PIL import Image
import zipfile
import csv
import face_recognition

bd_configPath = 'models/bib_detector/RBNR2_custom-yolov4-tiny-detector.cfg'
bd_weightsPath = 'models/bib_detector/RBNR2_custom-yolov4-tiny-detector_best.weights'
bd_classes = ['bib']
bd = BibDetector(bd_configPath, bd_weightsPath, bd_classes)
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

# Loop over all photos in the download directory
error = 0
participants = {}
unidentified_participants = {}

with open("results_error.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["filename", "bib_number", "error"])

    for i, filename in enumerate(os.listdir(extract_dir)):

        filenumber = i + 1
        human = 0
        full_image = False
        bib = 0

        try:
            try:
                # Load the photo and convert it to grayscale
                img = cv2.cvtColor(cv2.imread(os.path.join(extract_dir, filename)), cv2.COLOR_BGR2RGB)
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
                # print(f"[ERROR] {filenumber} {filename} couldn't detect any human:", err)
                continue

            if len(results) == 0:
                error += 1
                # print(f"[ERROR] {filenumber} {filename} detected nothing.")
                continue

            for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):

                bib_detections_probables = bd.detect(img, 0.25)

                if human_model.config.id2label[label.item()] == "person" and round(score.item(), 3) > 0.9:
                    body_box = [int(i) for i in box.tolist()]
                    human += 1

                    runner = Runner(filename=filename, img=img)
                    runner.body_location = body_box                                                                      # Save the body box of the runner
                    runner.detect_face()                                                                                 # Detects faces in the body box
                    runner.embeddings()                                                                                  # Calculate the embedding vector for the detected faces

                    bib_detections = []

                    for bib_box in bib_detections_probables:
                        (x, y, w, h) = bib_box[1]

                        if x >= body_box[0] and y >= body_box[1] and x + w <= body_box[2] and y + h <= body_box[3]:
                            bib_detections.append(['bib', [x, y, w, h]])

                    if len(bib_detections):

                        for bib_box in bib_detections:
                            (x1, y1, x2, y2) = convert_opencv_to_dlib(bib_box[1])
                            runner.bib_location = (x1, y1, x2, y2)

                            bib_number = bd.process_image(img[y1:y2, x1:x2])

                            if is_correct(bib_number, event="TBT"):
                                bib += 1
                                runner.bib_number = bib_number
                                runner.identified = True

                                if str(bib_number) not in participants.keys():
                                    participants[str(bib_number)] = Participant(bib_number=bib_number)

                                participants[str(bib_number)].add_new_sample(filename, runner=runner)

                                save_photo(organize_dir, extract_dir, bib_number, filename)

                                csvwriter.writerow([filename, bib_number, None])

                            else:
                                if filename in unidentified_participants.keys():
                                    unidentified_participants[filename] = list()

                                unidentified_participants[filename].append(runner)
                                # print(f"[ERROR] {filenumber} {filename} invalid bib number range:", bib_number)
                                continue
                    else:
                        if filename in unidentified_participants.keys():
                            unidentified_participants[filename] = list()

                        unidentified_participants[filename].append(runner)
                        # print(f"[ERROR] {filenumber} {filename} no bib detected.")
                        continue

            if human == 0 or bib == 0:
                error += 1
                err = f"[ERROR] {filenumber} {filename} no human or bib detected. humans: {human} & bibs: {bib}"
                csvwriter.writerow([filename, "", err])
                print(err)
                continue

        except Exception as err:
            error += 1
            csvwriter.writerow([filename, "", err])
            print("[ERROR]", filenumber, filename, err)
            continue

# Use Face features to match unidentified images
