import pandas as pd
import os
from tqdm.auto import tqdm
from config import *
tqdm.pandas()
import json
import re
from go_tree_sitter.go_parser import GoParser
from go_tree_sitter.go_tree_sitter_tool import GoTreeSitterTool
from tree import ASTNode
import pickle
from gensim.models.callbacks import CallbackAny2Vec


class Pipeline:

    """Pipeline class

    Args:
        root (str): Path to the folder containing the data.
    """

    def __init__(self, root: str):
        self.root = root
        self.ratio = RATIO
        self.size = None

        self.sources = None # ast集合
        self.train_data = None
        self.dev_data = None
        self.test_data = None

    # parse source code
    def get_parsed_source(self, input_file: str, output_file: str = None) -> pd.DataFrame:

        """Parse Go code using go-tree-sitter

        The method reads the input_file and generate a 'output_file', which is a 
        DataFrame containing the columns  code (go code parsed by go-tree-sitter). 

        Args:
            input_file (str): Path to the input file
            output_file (str): Path to the output file


        Returns:
            pd.DataFrame: DataFrame with the column code (Go code parsed by
            go-tree-sitter).
        """

        input_file_path = os.path.join(self.root, input_file)
        
        processed_code = pd.read_pickle(input_file_path)

        # 使用go-tree-sitter生成code的ast
        parser = GoParser()
        processed_code['code'] = processed_code['code'].progress_apply(parser.parse) # 每一个值都是一个tree_sitter.Tree Obeject
        # processed_code.to_pickle(os.path.join(self.root, output_file))
        self.sources = processed_code

        return self.sources
    

    # split data for training, developing and testing
    # 这里我可以全部将数据存储进训练集里
    def split_data(self):
        data = self.sources.copy()
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')] # self.ratio是一个形如'8:1:1'的字符串，表示训练集、验证集和测试集的比例。
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)
        data = data.sample(frac=1, random_state=666) #打乱数据
        self.train_data = data.iloc[:train_split]
        self.dev_data = data.iloc[train_split:val_split]
        self.test_data = data.iloc[val_split:]

    # node是一个string类的代表ast的字符串
    def get_sequences(self, node, sequence):
        current = ASTNode(node)
        if current.token is not '':
            sequence.append(current.token)
        for child in node.children:
            self.get_sequences(child, sequence) # 深度优先地遍历整个 AST 树
        # TODO: 这里什么时候加入"End" --> 已解决
        if current.is_compound:
            sequence.append('End')



    # 转化成ST-tree sequence
    def trans_to_sequences(self,ast):
        sequence = []
        self.get_sequences(ast.root_node, sequence)
        return sequence


    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file=None):
        
        trees = self.train_data.copy() # 深拷贝
        if not os.path.exists(self.root+'train/embedding'):
            os.mkdir(self.root+'train/embedding')
 

        corpus = trees['code'].progress_apply(self.trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        trees.to_csv(self.root+'train/programs_ns.tsv')

        # 调用word2vec
        from gensim.models.word2vec import Word2Vec

        # 加个进度条吧～
        class ProgressCallback(CallbackAny2Vec):
            def __init__(self, total_epochs):
                self.total_epochs = total_epochs
                self.progress_bar = tqdm(total=total_epochs, desc='Training Progress')

            def on_epoch_end(self, model):
                self.progress_bar.update(1)

            def on_train_end(self, model):
                self.progress_bar.close()
        progress_callback = ProgressCallback(W2V_EPOCH)
        
        
        w2v = Word2Vec(sentences=corpus,vector_size=EMBEDDING_SIZE, epochs=W2V_EPOCH, workers=16, sg=1, min_count=MIN_COUNT, max_final_vocab=VOCAB_SIZE, callbacks=[progress_callback])


        w2v.save(self.root+'train/embedding/node_w2v_' + str(EMBEDDING_SIZE))
        self.embedding=w2v


    def get_blocks(self, node, block_seq):
        children = node.children
        name = node.text.decode('utf-8')

        keywords = ['func', 'for', 'if']

        if children == []: #本身就是一个子节点
            block_seq.append(ASTNode(node))

        elif any(keyword in name for keyword in keywords):
            block_seq.append(ASTNode(node))

            # TODO: 这段逻辑？
            #if for is not in name:
            #    skip = 1
            #else:
            #    skip = len(children) - 1

            for i in range(len(children)):
                child = children[i]
                if not any(keyword in name for keyword in keywords):
                    block_seq.append(ASTNode(child))
                self.get_blocks(child, block_seq)
        elif '{' in name:
            block_seq.append(ASTNode(node))
            for child in node.children:
                if not any(keyword in name for keyword in keywords):
                    block_seq.append(ASTNode(child))
                self.get_blocks(child, block_seq)
            block_seq.append(ASTNode('End'))
        else:
            for child in node.children:
                self.get_blocks(child, block_seq)

    # generate block sequences with index representations
    def generate_block_seqs(self, data,part):
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.root+'train/embedding/node_w2v_' + str(EMBEDDING_SIZE)).wv

        max_token = word2vec.vectors.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [word2vec.key_to_index[token] if token in word2vec.key_to_index else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r): #传入的是一个Tree_sitter. Tree Obeject
            blocks = []
            self.get_blocks(r.root_node, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        data['code'] = data['code'].progress_apply(trans2seq)

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)
        
        check_or_create(self.root+part)
        data.to_pickle(self.root+part+'blocks.pkl')

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.get_parsed_source(input_file='programs.pkl')
        print('split data...')
        self.split_data()
        print('train word embedding...')
        self.dictionary_and_embedding(None)
        print('generate train block sequences...')
        self.generate_block_seqs(self.train_data, 'train/')
        print('generate dev block sequences...')
        self.generate_block_seqs(self.dev_data, 'dev/')
        print('generate test block sequences...')
        self.generate_block_seqs(self.test_data, 'test/')

ppl = Pipeline( './data/')
ppl.run()
