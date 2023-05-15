import os
import cv2
from PIL import Image
import json


def log(statement):
    print(statement)


def generate_annotations(file_name, label_json_path, output_folder):
    word_to_labels = json.load(
        open(os.path.join(label_json_path, "word_to_label.json"), "r")
    )
    annotations = []
    for word in os.listdir(output_folder):
        if word.endswith(".txt"):
            continue
        for video in os.listdir(os.path.join(output_folder, word)):
            video_id = video.split(".")[0]
            video_path = os.path.join(output_folder, word, video_id)
            annotations.append(
                f"{word}/{video_id} 1 {len(os.listdir(video_path))-1} {word_to_labels[word]}\n"
            )
    output_file = open(os.path.join(output_folder, f"{file_name}.txt"), "w+")
    output_file.writelines(annotations)
    output_file.close()


def double_frames(frame_list):
    return [val for val in frame_list for _ in range(2)]


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_frames(word, video, data_folder):
    vidcap = cv2.VideoCapture(os.path.join(data_folder, word, video))
    success, image = vidcap.read()
    frames = []
    while success:
        frames.append(image)
        success, image = vidcap.read()
    return frames


def resize_and_output(word, video, frames, output_folder, output_size=(200, 200)):
    for index, image in enumerate(frames):
        im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize(
            output_size, Image.ANTIALIAS
        )
        im.save(
            os.path.join(output_folder, word, video, f"img_{index:03d}.jpg"),
            "JPEG",
            quality=90,
        )


def convert_known_videos_to_frames(data_folder: str = "sample_test_data"):
    output_folder = f"{data_folder}/../processed_data"
    label_json_path = "utils/labels"
    MIN_FRAMES = 40

    words = os.listdir(data_folder)
    create_folder(output_folder)
    # iterate through every word
    for word in words:
        # create folder for word
        create_folder(os.path.join(output_folder, word))
        # iterate through every video for each word
        videos = os.listdir(os.path.join(data_folder, word))
        for video in videos:
            log(f"Processing {word}/{video}")

            video_id = video.split(".")[0]  # remove extension
            # create folder for individual video
            create_folder(os.path.join(output_folder, word, video_id))
            # generate frames
            frames = get_frames(word, video, data_folder)
            # duplicate frames if you have less than MIN_FRAMES frames
            while len(frames) <= MIN_FRAMES:
                frames = double_frames(frames)

            # save each frame in the folder as a 200x200 image
            resize_and_output(word, video_id, frames, output_folder)

    # specify file name without extension for annotations
    generate_annotations("annotations", label_json_path, output_folder)


def convert_unknown_videos_to_frames(data_folder: str = "sample_test_data"):
    output_folder = f"{data_folder}/../processed_data"
    label_json_path = "utils/labels"
    MIN_FRAMES = 40

    words = os.listdir(data_folder)
    create_folder(output_folder)
    # iterate through every word
    for word in words:
        # create folder for word
        create_folder(os.path.join(output_folder, word))
        # iterate through every video for each word
        videos = os.listdir(os.path.join(data_folder, word))
        for video in videos:
            log(f"Processing {word}/{video}")

            video_id = video.split(".")[0]  # remove extension
            # create folder for individual video
            create_folder(os.path.join(output_folder, word, video_id))
            # generate frames
            frames = get_frames(word, video, data_folder)
            # duplicate frames if you have less than MIN_FRAMES frames
            while len(frames) <= MIN_FRAMES:
                frames = double_frames(frames)

            # save each frame in the folder as a 200x200 image
            resize_and_output(word, video_id, frames, output_folder)

    # specify file name without extension for annotations
    generate_annotations("annotations", label_json_path, output_folder)
