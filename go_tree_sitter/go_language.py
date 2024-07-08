from tree_sitter import Language


class GoLanguage:
    PATH = "/bigdata/qiuhan/generate_go_code/data_processing/resources/build/my-languages.so"

    language = Language(PATH, "go")

    @staticmethod
    def use_query(query, node):
        return GoLanguage.language.query(query).captures(node)
