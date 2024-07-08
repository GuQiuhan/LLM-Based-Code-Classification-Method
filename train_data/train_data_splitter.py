from tqdm import tqdm

from data_io.file_io import data_from_jsonl, data_to_jsonl_append


class TrainDataSplitter:
    def __init__(self, src_path, dst_path, train_per):
        self.src_path = src_path
        self.dst_path = dst_path

        self.train_per = train_per

    def split(self):
        print(f"{'=' * 20} start splitting train data {'=' * 20}")

        data_num = len(list(data_from_jsonl(self.src_path)))
        train_num = int(data_num * self.train_per)
        test_dev_num = int((data_num - train_num) / 2)

        for i, src_line in tqdm(enumerate(data_from_jsonl(self.src_path))):
            if i < train_num:
                data_to_jsonl_append(self.dst_path + "/train.jsonl", src_line)
            elif i < train_num + test_dev_num:
                data_to_jsonl_append(self.dst_path + "/test.jsonl", src_line)
            else:
                data_to_jsonl_append(self.dst_path + "/dev.jsonl", src_line)
