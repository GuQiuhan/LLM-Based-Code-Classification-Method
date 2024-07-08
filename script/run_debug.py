import os

import torch
from transformers import RobertaTokenizer

from go_generate.go_generator import GoGenerator
from go_generate.test_case_maker import TestCaseMaker

MODEL_PATH = "PPY039/codet5-small-go_generation_v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = "D:\huggingface_cache"

go_generator = GoGenerator(MODEL_PATH, DEVICE, CACHE_DIR)

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)

DATA_COUNT = 71421
SRC_PATH = f"./output/split_train_data/data_{DATA_COUNT}/test.jsonl"
DST_PATH = f"./output/test_case/data_{DATA_COUNT}.jsonl"

if os.path.isfile(DST_PATH):
    os.remove(DST_PATH)

test_case_maker = TestCaseMaker(SRC_PATH, DST_PATH, go_generator, tokenizer)

test_case_maker.init_origin_data_list()
total_test_case_count = test_case_maker.make_test_case_loop(20)

print(f"{total_test_case_count}")
