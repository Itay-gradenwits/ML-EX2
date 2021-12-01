import numpy as np

def loss(x, y, w, b):
    print(w,b,w[0], b[0])
    loss = 0
    for i in range(len(w)):
        if(i != y):
            arg2 = np.float_(1 - np.dot(w[y], x) - np.dot(w[i], x) - b[y] - b[i])
            loss = max(loss, arg2)
    return loss


def test():
    arrays = [np.asarray([1,1,1]), np.asarray([2,2,2]), np.asarray([3,3,3])]
    print(loss([1,1,1], 0, arrays, [0,0,0]))

test()