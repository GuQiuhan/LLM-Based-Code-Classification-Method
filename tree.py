
class ASTNode(object):
    def __init__(self, node):
        self.node = node # ast字符串
        self.is_str = isinstance(self.node, str) # 检查是否是str类型
        self.token = self.get_token()
        self.children = self.add_children()
        self.is_compound =self.is_compound()
        
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
        return [ASTNode(child) for child in children]

    def is_compound(self): #用于build block，判断这个节点是不是一个block的起始点
        return self.token=="{"


