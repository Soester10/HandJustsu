import os
from PIL import Image
import threading

path = "/../../processed_frames_dataset"
path2 = "/../../resized_frames_dataset"
dir = os.getcwd() + path
dir2 = os.getcwd() + path2
# count = 0


def process_word_list(word_list):
    thread_id = threading.get_ident()
    count = 0
    for word in word_list:
        count += 1
        videos_path = f'{dir}/{word}'
        print(f'Processing word {count}: {word}')
        if os.path.isdir(videos_path):
            resized_video_path = f'{dir2}/{word}'
            os.mkdir(resized_video_path)
            for video_id in os.listdir(videos_path):
                print(f'{thread_id}::: {count}: {word}/{video_id}')
                single_video_path = f'{videos_path}/{video_id}/'
                resized_single_video_path = f'{resized_video_path}/{video_id}/'
                os.mkdir(resized_single_video_path)
                image_list = os.listdir(single_video_path)
                for item in image_list:
                    im = Image.open(single_video_path+item)
                    f, e = os.path.splitext(single_video_path+item)
                    imResize = im.resize((200, 200), Image.ANTIALIAS)
                    imResize.save(resized_single_video_path +
                                  f'{item}', 'JPEG', quality=90)


threads = []

subfolders = os.listdir(dir)
subfolders.remove("annotations.txt")

for i in range(0, 2000, 400):
    subarray = subfolders[i: i + 400]
    thread = threading.Thread(target=process_word_list, args=(subarray,))
    threads.append(thread)

for thread in threads:
    thread.start()

    # Wait for all the threads to finish
for thread in threads:
    thread.join()
