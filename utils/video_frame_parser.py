import json
import os
import cv2

PROCESSED_FRAMES_PATH = '../processed_frames_dataset/'


if not os.path.exists(PROCESSED_FRAMES_PATH):
    os.makedirs(PROCESSED_FRAMES_PATH)

missing_videos = set()

with open("../wlasl_dataset/missing.txt", "r") as f:
    missing_videos_list = f.readlines()
    for id in missing_videos_list:
        missing_videos.add(id.strip())

annotations_file = open(f"{PROCESSED_FRAMES_PATH}annotations.txt", "a")


with open('../wlasl_dataset/WLASL_v0.3.json') as json_file:
    data = json.load(json_file)
    label_id = 0
    for term in data:
        word = term["gloss"]
        if not os.path.exists(f"{PROCESSED_FRAMES_PATH}{word}/"):
            os.makedirs(f"{PROCESSED_FRAMES_PATH}{word}/")

        video_list = term["instances"]
        for video in video_list:
            video_id = video["video_id"]
            if video_id not in missing_videos:
                bbox = video["bbox"]

                if not os.path.exists(f"{PROCESSED_FRAMES_PATH}{word}/{video_id}/"):
                    os.makedirs(f"{PROCESSED_FRAMES_PATH}{word}/{video_id}/")

                vidcap = cv2.VideoCapture(
                    f'../wlasl_dataset/videos/{video_id}.mp4')
                success, image = vidcap.read()
                start_frame = video["frame_start"]
                end_frame = video["frame_end"] if video["frame_end"] > 0 else 100000
                frameIndex = 1
                index = 1
                print(f'Processing {label_id} {word} {video_id}')
                while success:
                    if start_frame <= frameIndex <= end_frame:
                        # save frame as JPEG file
                        cv2.imwrite(
                            f"{PROCESSED_FRAMES_PATH}{word}/{video_id}/img_{index:03}.jpg", image)
                        index += 1
                    success, image = vidcap.read()
                    frameIndex += 1
                # print(f"{word}/{video_id} 1 {index - 1} {label_id}")
                annotations_file.write(
                    f"{word}/{video_id} 1 {index - 1} {label_id}\n")
        label_id += 1
    annotations_file.close()
