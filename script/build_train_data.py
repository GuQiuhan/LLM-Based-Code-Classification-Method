import os

from transformers import RobertaTokenizer

from train_data.train_data_builder import TrainDataBuilder

DATA_COUNT = 71421

SRC_PATH = f"./output/processed_data/data_{DATA_COUNT}.jsonl"
DST_PATH = f"./output/train_data/data_{DATA_COUNT}.jsonl"

if os.path.isfile(DST_PATH):
    os.remove(DST_PATH)

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small', cache_dir="D:\huggingface_cache")

train_data_builder = TrainDataBuilder(SRC_PATH, DST_PATH, tokenizer)
total_count = train_data_builder.build()

print(f"{total_count}")
