from tree_sitter import Language

Language.build_library(
  # Store the library in the `build` directory
  '/bigdata/qiuhan/generate_go_code/data_processing/resources/build/my-languages.so',

  # Include one or more languages
  [
    '/bigdata/qiuhan/generate_go_code/data_processing/resources/tree-sitter-go-master'
  ]
)
