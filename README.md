# HandJutsu (NYU-CSGY6953-DL-Project)

## Description
This project aims to classify hand gestures into meaningful words, to help understand and interpret sign language. The goal is to develop a deep learning model capable of detecting sign language gestures and translating them into text. Sign language serves as a vital means of communication for individuals with speech and/or hearing impairments, making accurate sign language detection systems essential for facilitating understanding and effective communication. Despite the challenges posed by the complexity and variability of continuous hand and finger motions, this project aims to bridge the gap by leveraging state-of-the-art computer vision models to accurately translate sequences of gestures into meaningful text. By achieving accurate and reliable sign language detection, this project aims to enhance inclusivity and communication accessibility for individuals with speech and/or hearing impairments. To achieve our goal, we have used the WLASL dataset from [kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed).

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
├── output
│   └── output.txt
├── checkpoint
│   └── ckpt.pth
└── utils
    ├── HandJutsu.pth
    ├── custom_dataloader
    │   ├── custom_dataloader.py
    │   └── dataloader_main.py
    ├── graph_plotters
    │   ├── frames_plotters.py
    │   └── metrics_plotters.py
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
| --scratch | trains from scratch without checkpoint, uses checkpoint by default (if available) | bool: False |
| --optimizer | decide optimizer for training model | str: adadelta |
| --lr | learning rate for training | float: 0.1 |
| --batch_size | batch size to be trained with | int: 32 |
| --optimal_batch_size | optimal batch size supported by hardware | int: 8 |
| --no_batch_mul | if a batch multiplier need not be used, bacth mul used by default | bool: False |
| --plot_result | to plot test and train accuracies and losses | bool: False |


## Output
Outputs written to
```
output/output.txt
```

## Note
> - Dataset has not been uploaded due to size limit, can be downloaded from [here](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) <br>
> - Download our best checkpoint from [here](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) <br>
> - Follow code structure for running the code and adding checkpoint <br>
