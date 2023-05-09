import os
import cv2
from PIL import Image
import json

dirname = os.path.dirname(__file__)
data_folder = os.path.join(dirname, "../sample_test_data")
output_folder = os.path.join(dirname, "../processed_sample_data")
label_json_path = os.path.join(dirname, "./labels")
MIN_FRAMES = 40

ENABLE_LOGS = True


def log(statement):
    if ENABLE_LOGS:
        print(statement)


def generate_annotations(file_name):
    word_to_labels = json.load(
        open(os.path.join(label_json_path, 'word_to_label.json'), "r"))
    annotations = []
    for word in os.listdir(output_folder):
        if word.endswith(".txt"):
            continue
        for video in os.listdir(os.path.join(output_folder, word)):
            video_id = video.split(".")[0]
            video_path = os.path.join(output_folder, word, video_id)
            annotations.append(
                f'{word}/{video_id} 1 {len(os.listdir(video_path))} {word_to_labels[word]}\n')
    output_file = open(os.path.join(output_folder, f'{file_name}.txt'), "w+")
    output_file.writelines(annotations)
    output_file.close()


def double_frames(frame_list):
    return [val for val in frame_list for _ in range(2)]


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_frames(word, video):
    vidcap = cv2.VideoCapture(os.path.join(data_folder, word, video))
    success, image = vidcap.read()
    frames = []
    while success:
        frames.append(image)
        success, image = vidcap.read()
    return frames


def resize_and_output(word, video, frames, output_size=(200, 200)):
    for index, image in enumerate(frames):
        im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize(
            output_size, Image.ANTIALIAS)
        im.save(os.path.join(output_folder, word, video, f'img_{index:03d}.jpg'),
                "JPEG", quality=90)


if __name__ == "__main__":

    create_folder(output_folder)  # create root output folder
    # iterate through every word
    for word in os.listdir(data_folder):
        # create folder for word
        create_folder(os.path.join(output_folder, word))
        # iterate through every video for each word
        for video in os.listdir(os.path.join(data_folder, word)):

            log(f"Processing {word}/{video}")

            video_id = video.split(".")[0]  # remove extension
            # create folder for individual video
            create_folder(os.path.join(output_folder, word, video_id))
            # generate frames
            frames = get_frames(word, video)
            # duplicate frames if you have less than MIN_FRAMES frames
            while len(frames) <= MIN_FRAMES:
                frames = double_frames(frames)

            # save each frame in the folder as a 200x200 image
            resize_and_output(word, video_id, frames)

    # specify file name without extension for annotations
    generate_annotations("annotations")
