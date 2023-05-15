# HandJutsu (NYU-CSGY6953-DL-Project)

## Description
This project aims to classify hand gestures into meaningful words, to help understand and interpret sign language.

## Setup (Ideally use Python 3.8 +)
Setup virtual env
```
$ python3 -m venv python_venv
```

For Mac lol/ Linux
```
$ source python_venv/bin/activate
```

For Windows
```
$ python_venv/Script/activate
```

Download and install Cuda 11.7 from [here](https://pytorch.org/get-started/locally/)

Install Dependencies
```
$ python -m pip install -r requirements.txt
```

## Code Structure
```
├── README.md
├── main.py
├── models
│   └── VisionTransformer.py
├── requirement.txt
├── data (example)
│   └── hello
│       └── 01217.mp4
├── test_data (example)
│   └── sample_test_data
│       ├── careful
│       │   └── 09188.mp4
│       └── order
│           └── 40183.mp4
└── utils
    ├── HandJutsu.pth
    ├── custom_dataloader
    │   ├── custom_dataloader.py
    │   └── dataloader_main.py
    ├── custom_test.py
    ├── dataloader.py
    ├── dataset_parser
    │   ├── image_duplication.py
    │   ├── image_resize.py
    │   └── video_frame_parser.py
    ├── labels
    │   ├── label_to_word.json
    │   └── word_to_label.json
    ├── optimizers.py
    ├── process_videos_to_format.py
    ├── test.py
    └── train.py
```
    
## Commands
To train with default values (trains from checkpoint if ckpt.pth is present)
```
$ python main.py
```

### Arugments
| Args | Desc | Default |
|---|---|---|
| --test | skips training and only tests the model with valid checkpoint | bool: False |
| --test_data_path | path to videos for testing | str: ./test_data/sample_test_data |
| --unlabeled_test | to know determine the data structure, words/videos or videos | bool: False |
| --epochs | number of epochs for training | int: 100 |
| --scratch | trains from scratch without checkpoint | bool: False |


## Output
Outputs written to
```
output/output.txt
```