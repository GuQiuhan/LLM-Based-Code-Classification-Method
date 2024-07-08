from tree_sitter import Language


class GoLanguage:
    PATH = "/bigdata/qiuhan/cluster/astnn/resources/build/my-languages.so"
    

    language = Language(PATH, "go")

    @staticmethod
    def use_query(query, node):
        return GoLanguage.language.query(query).captures(node)
