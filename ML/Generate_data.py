import numpy as np
from LSR import LR_LS
#生成随机数
np.random.seed(1234)
x = np.random.rand(500,3)
#构建映射关系，模拟真实的数据待预测值,映射关系为y = 3.1 + 5.2*x1 + 11.8*x2，可自行设置值进行尝试
y = x.dot(np.array([3.1,5.2,11.8]))

if __name__ == "__main__":
    print("Least squares revisited:")
    lr_ls = LR_LS()
    lr_ls.fit(x,y)
    print("估计的参数值：%s" %(lr_ls.w))
    x_test = np.array([3,7,9]).reshape(1,-1)
    print("预测值为: %s" %(lr_ls.predict(x_test)))

