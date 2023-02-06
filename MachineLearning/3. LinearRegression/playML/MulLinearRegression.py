import numpy as np
from .metrics import r2_score

class MulLinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None # 系数
        self.interception_ = None #截距
        self._theta = None # 整体

    def fit_normal(self, X_train, y_train):
        """根据训练数据集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 在原来的训练集前面补上一列
        Xb = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(Xb.T.dot(Xb)).dot(Xb.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:] 
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        
        Xb = np.hstack([np.ones((len(X_predict),1)), X_predict])
        return Xb.dot(self._theta)

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "MulLinearRegression()"