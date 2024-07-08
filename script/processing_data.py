import os

from data_processor.data_processor import DataProcessor

DATA_COUNT = 71421

SRC_PATH = f"./output/raw_data/data_{DATA_COUNT}.jsonl"
DST_PATH = f"./output/processed_data/data_{DATA_COUNT}.jsonl"

if os.path.isfile(DST_PATH):
    os.remove(DST_PATH)

data_processor = DataProcessor(SRC_PATH, DST_PATH)
total_count = data_processor.process()

print(f"{total_count}")
