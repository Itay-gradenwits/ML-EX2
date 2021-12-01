import numpy as np

def test():
    arrays = [np.asarray([1,1,1]), np.asarray([2,2,2]), np.asarray([3,3,3])]
    print(np.average(np.multiply(arrays, arrays)))

test()