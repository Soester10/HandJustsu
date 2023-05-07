import os
import shutil

# process all folders that have under 40 images to double the image count (doubling the frame rate)

path = "/../../processed_frames_dataset"
dir = os.getcwd() + path
count = 0

MIN_IMAGE_COUNT = 40

for word in os.listdir(dir):
    # print(word)
    videos_path = f'{dir}/{word}'
    if os.path.isdir(videos_path):
        for video_id in os.listdir(videos_path):
            # print(video_id)
            single_video_path = f'{videos_path}/{video_id}'
            image_list = os.listdir(single_video_path)
            if len(image_list) <= MIN_IMAGE_COUNT:
                print(count)
                count += 1
                target_file_count = 2 * len(image_list)
                for file in sorted(image_list, reverse=True):
                    # print(f'{videos_path}/{video_id}/{file}')
                    shutil.copy(f'{videos_path}/{video_id}/{file}',
                                f'{videos_path}/{video_id}/img_{target_file_count:03d}.jpg')
                    target_file_count -= 1
                    if target_file_count > 1:
                        shutil.copy(f'{videos_path}/{video_id}/{file}',
                                    f'{videos_path}/{video_id}/img_{target_file_count:03d}.jpg')
                    target_file_count -= 1
