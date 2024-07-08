import os

from tqdm import tqdm

from data_io.file_io import data_to_jsonl_append
from dataset_loader.dataset_loader import DatasetLoader

DATASET_PATH = "codeparrot/github-code-clean"
LANGUAGES = ["GO"]
SPLIT = "train"
STREAMING = True
CACHE_DIR = "D:\\huggingface_cache"

dataset_loader = DatasetLoader(is_iter=STREAMING, path=DATASET_PATH, languages=LANGUAGES, split=SPLIT,
                               streaming=STREAMING, cache_dir=CACHE_DIR)

DATA_COUNT = 71421
SAVED_PATH = f"./output/raw_data/data_{DATA_COUNT}.jsonl"

if os.path.isfile(SAVED_PATH):
    os.remove(SAVED_PATH)

for i in tqdm(range(DATA_COUNT)):
    data = next(dataset_loader.dataset)
    data_to_jsonl_append(SAVED_PATH, data)
