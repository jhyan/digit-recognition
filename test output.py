import numpy as np
res = np.array([1,2,3,4])
# np.savetxt("prediction.txt", res, fmt="%d")

col1 = np.asarray(["id"] + list(range(10)))
col2 = np.arange(11)
print(col1.shape)
print(col2.shape)
res = np.concatenate((col1, col2),axis=1)
print (res)

np.savetxt("foo.csv", res, fmt="%s", delimiter=",")