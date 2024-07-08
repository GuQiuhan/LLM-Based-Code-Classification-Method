from tqdm import tqdm

from data_io.file_io import *
from filter.internal_import_filter import InternalImportFilter
from go_tree_sitter.go_parser import GoParser
from go_tree_sitter.go_tree_sitter_tool import GoTreeSitterTool
from filter.undefined_behavior_filter import UndefinedBehaviorFilter


class DataProcessor:

    def __init__(self, src_path, dst_path):
        self.src_path = src_path
        self.dst_path = dst_path

        self.parser = GoParser()

        self.MAX_SIZE = 10000

    def do_filter(self, node, size):
        return size > self.MAX_SIZE or \
            GoTreeSitterTool.has_error(node) or \
            InternalImportFilter.do_filter(node) or \
            UndefinedBehaviorFilter.do_filter(node) or \
            False

    def do_filter_count(self, node, size):
        flag = False
        count = {"over_size": 0, "has_error": 0, "internal_import": 0, "undefined_behavior": 0}
        if size > self.MAX_SIZE:
            flag = True
            count["over_size"] += 1
        if GoTreeSitterTool.has_error(node):
            flag = True
            count["has_error"] += 1
        if InternalImportFilter.do_filter(node):
            flag = True
            count["internal_import"] += 1
        if UndefinedBehaviorFilter.do_filter(node):
            flag = True
            count["undefined_behavior"] += 1
        return flag, count

    def delete_all_comment(self, code):
        node = self.parser.parse(code)

        comments = [elem.text.decode("utf8") for elem in GoTreeSitterTool.get_comment(node)]
        comments.sort(key=lambda x: len(x), reverse=True)

        for comment in comments:
            code = code.replace(comment, "")

        rst_code = ""
        for line in code.splitlines():
            if len(line) == 0:
                rst_code = rst_code + line + "\n"
            if len(line.replace("\t", "")) != 0:
                rst_code = rst_code + line + "\n"

        return rst_code.strip()

    def process(self):
        print(f"{'=' * 20} start processing raw data {'=' * 20}")
        total_count = {"pass": 0, "over_size": 0, "has_error": 0, "internal_import": 0, "undefined_behavior": 0}
        for src_line in tqdm(data_from_jsonl(self.src_path)):
            src_code = self.delete_all_comment(src_line["code"])
            root_node = self.parser.parse(src_code)

            src_size = len(src_code.encode("utf8"))

            flag, count = self.do_filter_count(root_node, src_size)
            if not flag:
                total_count["pass"] += 1
                dst_line = {"code": src_code, "size": src_size}
                data_to_jsonl_append(self.dst_path, dst_line)
            else:
                total_count["over_size"] += count["over_size"]
                total_count["has_error"] += count["has_error"]
                total_count["internal_import"] += count["internal_import"]
                total_count["undefined_behavior"] += count["undefined_behavior"]

        return total_count
