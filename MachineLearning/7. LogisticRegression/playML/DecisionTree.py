
'''
定义一个节点类，每个节点代表着一个样本集合的划分。
data = 样本特征
label = 样本标签
dim = 在哪一维度上划分
value = 在 dim 维度上的划分边界
'''
class Node:
    def __init__(self, data, label, dim, value):
        self.data = x_data
        self.label = y_label
        self.dim = dim
        self.value = value
        self.left = None
        self.right = None

'''
定义一个决策树学习机，
开局一个根模型全靠训练。
'''
class DecisionTree:
    def __init__(self):
        self.root = None

    def fit(self, x_train, y_train):
        
        #传入某一样本子集的标签，计算信息熵
        def entropy(label) :
            cnt = Counter(label)
            ent = 0.
            for val, num in cnt :
                p = num / len(label)
                ent += -p * log(p)
            return ent

        def split(data, label, dim, val) :
            idx_l = data[:, dim] <= val
            idx_r = data[:, dim] > val
            return data[idx_l], data[idx_r], data[idx_l], data[idx_r]

        # 单次划分数据集
        def try_split(data, label) :
            BestEntropy = float("inf")
            BestDim, BestVal = -1, -1
            
            # 选取一个维度进行划分
            for d in range(data.shape[1]) :
                SortIdx = np.argsort(data[:, d])
                SortData = data[SortIdx, d]

                # 枚举最优划分属性
                for i in range(1, len(SortData)) :
                    if SortData[i-1] == SortData[i] : continue
                    val = (SortData[i-1] + SortData[i]) / 2
                    x_l, x_r, y_l, y_r = split(data, label, d, val)

                    # 计算最优信息熵
                    p_l = len(x_l) / len(data)
                    p_r = len(x_r) / len(data)
                    ent = p_l * entropy(y_l) + p_r * entropy(y_r)
                    if ent < BestEntropy :
                        BestEntropy = ent
                        BestDim, BestVal = d, val
            
            return BestEntropy, BestDim, BestVal
        
        # 利用西瓜书上的算法生成决策树
        def build_tree(data, label, eps=1e-7) :

            # 选择最优划分属性
            ent, dim, val = try_split(data, label)
            node = Node(data, label, dim, value)
            
            # 如果 D 的信息熵小于阈值
            if ent < eps : 
                return node

            x_l, x_r, y_l, y_r = split(data, label, dim, val)
            node.left = build_tree(x_l, y_l)
            node.right = build_tree(x_r, y_r)
            return node

        self.root = build_tree(x_train, y_train)
        return self

    def predict(self, x_pred):
        
        # 递归遍历决策树，寻找对应分类
        def dfs(x_data, p) :
            pred = -1
            if x_data[p.dim] <= p.val and p.left :
                pred = dfs(x_data, p.left)
            elif x_data[p.dim] > p.val and p.right :
                pred = dfs(x_data, p.right)
            else :
                cnt = Counter(p.label)
                pred = cnt.most_common(1)[0][0]
            return pred

        y_pred = []
        for x in x_pred :
            y_pred.append(dfs(x, self.root))

        return np.array(y_pred)

    # 模型预测准确值
    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return np.sum(y_predict == y_test) / len(y_predict)

    def __repr__(self):
        return "DTree(criterion='entropy')"