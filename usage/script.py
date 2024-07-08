import pandas as pd
from tqdm.auto import tqdm
from config import *
tqdm.pandas()
import torch
from go_tree_sitter.go_parser import GoParser
from model import BatchProgramClassifier
import numpy as np

class ASTNode(object):
    def __init__(self, node):
        self.node = node # ast字符串
        self.is_str = isinstance(self.node, str) # 检查是否是str类型
        self.token = self.get_token()
        self.children = self.add_children()

    def is_leaf(self):
        return self.node.child_count == 0

    def get_token(self, lower=True):
        if self.is_str:
            return self.node
        token=''
        if self.is_leaf():
            token = self.node.text.decode('utf-8')
            token = token.lower()
        return token

    def add_children(self):
        if self.is_str:
            return []
        children = self.node.children
        if self.token in ['FuncDef', 'If', 'While', 'DoWhile','Switch']:
            return [ASTNode(children[0][1])]
        elif self.token == 'For':
            return [ASTNode(children[c][1]) for c in range(0, len(children)-1)]
        else:
            return [ASTNode(child) for child in children]

    def compound(self): #用于build block，判断这个节点是不是一个block的起始点
        return self.token=="{"

class Pipeline:

    def __init__(self, s):
        self.original_program=s
        self.sources = None # ast
        self.input_data = None # block

    # parse source code
    def get_parsed_source(self) -> pd.DataFrame:
        processed_code = pd.DataFrame({'id': [0], 'code': [self.original_program], 'label': [0]}) 
        # 使用go-tree-sitter生成code的ast
        parser = GoParser()
        processed_code['code'] = processed_code['code'].apply(parser.parse) # 每一个值都是一个tree_sitter.Tree Obeject
        # processed_code.to_pickle(os.path.join(self.root, output_file))
        self.sources = processed_code

        return self.sources

    # node是一个string类的代表ast的字符串
    def get_sequences(self, node, sequence):
        current = ASTNode(node)
        if current.token is not '':
            sequence.append(current.token)
        for child in node.children:
            self.get_sequences(child, sequence) # 深度优先地遍历整个 AST 树
        if current.is_compound:
            sequence.append('End')

   
    def get_blocks(self, node, block_seq):
        children = node.children
        name = node.text.decode('utf-8')

        keywords = ['func', 'for', 'if']

        if children == []: #本身就是一个子节点
            block_seq.append(ASTNode(node))

        elif any(keyword in name for keyword in keywords):
            block_seq.append(ASTNode(node))

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
    def generate_block_seqs(self):
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load('embedding/node_w2v_128').wv # 直接加载，需要保证文件存在于当前路径下

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
        
        self.input_data=self.sources.copy()
        self.input_data['code'] = self.input_data['code'].apply(trans2seq)

    # run for processing data to train
    def run(self):
        #print('parse source code...')
        self.get_parsed_source()
        #print('generate training block sequences...')
        self.generate_block_seqs()
        
        tmp = self.input_data.iloc[0: 1]
        data = []
        for _, item in tmp.iterrows():
            data.append(item[1])
            
        return data
    
        #return self.input_data['code']
    

class Cluster:
    @classmethod
    def cluster(cls,program_string=None):

        if not isinstance(program_string, str):
            # 如果不是字符串类型，则返回
            return "Error: program_string should be a string type."
        
        model_input=Pipeline(program_string).run()
        

        # 调用word2vec
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load("embedding/node_w2v_128").wv
        embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
        embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

        MAX_TOKENS = word2vec.vectors.shape[0]
        EMBEDDING_DIM = word2vec.vectors.shape[1]

        # 加载模型，需要有config.py
        model=BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
        if USE_GPU:
            model.cuda()
        
        model.load_state_dict(torch.load('saved_model/best_model.bin'))  # 保证存在路径
        model.eval()  # 设置为评估模式，即不启用 Dropout 等训练时的特定操作

        with torch.no_grad():  # 禁用梯度计算,只进行推理
            output = model(model_input)  # 调用模型进行推理
        

        
        max_value, max_index = torch.max(output[0], dim=0)

        return max_index.item()+1,output[0].tolist()


''' 使用方法示例：
'''
input_string="package domain\n\n\nfunc MathPaymentMethodFlag(methods []int) int {\n\tf := 0\n\tfor _, v := range methods {\n\t\tf |= 1 << uint(v-1)\n\t}\n\treturn f\n}\n\n\n\n\nfunc AndPayMethod(payFlag int, method int) bool \n\tf := 1 << uint(method-1)\n\treturn payFlag&f == f\n"

label, probability=Cluster().cluster(input_string)

print("The category of the program is（1 for Normal Data, 2 for Anomalous Data）："+str(label))
print("Specific probability of 1:2: "+str(probability))


