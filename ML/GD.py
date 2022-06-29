import numpy as np
class LR_GD:
    def __init__(self):
        self.w=None
    def fit(self,X,y,alpha=0.02,con=1e-10):
        y=y.reshape(-1,1)
        #print(y)
        [m,d]=np.shape(X)
        self.w=np.zeros((d))#初始化
        tol = 1e5
        while tol > con:
            h_f = X.dot(self.w).reshape(-1,1) 
            theta = self.w + alpha*np.mean(X*(y - h_f),axis=0) #计算迭代的参数值，mean函数是求平均值，axis=0为求每列的平均值，返回1*n矩阵
            tol = np.sum(np.abs(theta - self.w))
            self.w = theta
    def predict(self, X):
        y_pred = X.dot(self.w)
        return y_pred  

