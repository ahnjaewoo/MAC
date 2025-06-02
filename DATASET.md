## Dataset Setup Instructions
The following instructions describe how to set up the COCO, MSR-VTT, and AudioCaps datasets.

### COCO
Run the following commands to download and extract the required files:
```bash
cd dataset_processing/data/COCO

# Download image data
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip

wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
```
Then, the directory should be structured as follows:
```
COCO/
├── annotations/
│   ├── captions_train2014.json
│   ├── captions_val2014.json
│   ├── instances_train2014.json
│   ├── instances_val2014.json
│   ├── karpathy_test_coco_ids.txt
│   ├── karpathy_val_coco_ids.txt
│   ├── person_keypoints_train2014.json
│   ├── person_keypoints_val2014.json
│   ├── pos_stats_test.json
│   └── pos_stats_train.json
├── train2014/
│   ├── COCO_train2014_000000000009.jpg
│   └── ...
└── val2014/
    ├── COCO_val2014_000000581929.jpg
    └── ...
```
---
### MSRVTT
Run the following commands to download and extract the required files:
```bash
cd dataset_processing/data/MSR-VTT
# Download the source videos from:
# https://www.mediafire.com/folder/h14iarbs62e7p/shared
```
Then, the directory should be structured as follows:
```
MSR-VTT/
├── retrieval_task/
│   ├── pos_stats_test.json
│   ├── pos_stats_train.json
│   ├── test_jsfusion.json
│   └── train.json
└── videos/
    ├── video0.mp4
    ├── ...
    └── video9999.mp4
```
---
### AudioCaps
Run the following commands to download and extract the required files:
```bash
cd dataset_processing/data/AudioCaps
# Audio files are hosted by the original authors.
# Please refer to the official dataset page:
# https://github.com/cdjkim/audiocaps/blob/master/dataset/README.md
```
Then, the directory should be structured as follows:
```
AudioCaps/
├── retrieval/
│   ├── pos_stats_test.json
│   ├── pos_stats_train.json
│   ├── retrieval_test.json
│   └── retrieval_train.json
└── audio/
    ├── __0Fp4K-2Ew_60.wav
    └── ...
```
---
### (Optional) Generating `pos_stats_*.json` Files

If you wish to generate the `pos_stats_*.json` files manually, you may use the following script:

```bash
python dataset_processing/preprocess_caption.py --dataset_name coco --data_split test
```
You may substitute `--dataset_name` with `msrvtt` or `audiocaps`, and `--data_split` with `train`, `val`, or `test` as needed.