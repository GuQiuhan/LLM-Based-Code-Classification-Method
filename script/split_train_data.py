import os

from train_data.train_data_splitter import TrainDataSplitter

DATA_COUNT = 71421

TRAIN_PER = 0.9

SRC_PATH = f"./output/train_data/data_{DATA_COUNT}.jsonl"
DST_PATH = f"./output/split_train_data/data_{DATA_COUNT}"

if os.path.isfile(DST_PATH + "/train.jsonl"):
    os.remove(DST_PATH + "/train.jsonl")
if os.path.isfile(DST_PATH + "/test.jsonl"):
    os.remove(DST_PATH + "/test.jsonl")
if os.path.isfile(DST_PATH + "/dev.jsonl"):
    os.remove(DST_PATH + "/dev.jsonl")

train_data_splitter = TrainDataSplitter(SRC_PATH, DST_PATH, TRAIN_PER)
train_data_splitter.split()
