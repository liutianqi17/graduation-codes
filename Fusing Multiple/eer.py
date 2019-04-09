import numpy as np
import matplotlib.pyplot as plt

path = r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\data\nuaa\ycbcr.txt'
data = np.loadtxt(path, dtype=float, delimiter=',', encoding='utf-8-sig')

TP = [0 for m in range(101)]
FP = [0 for n in range(101)]
v = 0
for i in np.arange(0, 1.01, 0.01):
    for j in range(3885):
        if data[j, 0] < i:
            if data[j, 1] == 1:
                TP[v] += 1
            else:
                FP[v] += 1
    v += 1

TPR = np.array(TP)/1911
FPR = np.array(FP)/1974

plt.plot(FPR, TPR)
plt.plot([0, 1], [1, 0])
plt.show()




