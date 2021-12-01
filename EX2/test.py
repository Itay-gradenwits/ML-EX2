import numpy as np
from numpy import linalg
import random
from numpy.lib.function_base import select

from numpy.random import shuffle

class preceptron:
    def __init__(self, train_x, train_y) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.w = np.zeros((3,5))
        self.bias = np.zeros(3)
        self.rate = 0.5
    
    def train(self):
        local_rate = self.rate
        train_x_local = self.normalize_data_set(self.train_x)
        train_y_local = np.copy(self.train_y)
        print(train_x_local, train_y_local)
        epochs = 300
        for e in range(epochs):
            count = 0
            self.shuffle(train_x_local, train_y_local)
            for x, y_iter in zip(train_x_local, train_y_local):
                y_hat = np.argmax(np.dot(self.w,x))
                y = int(y_iter[0])
                if(y_hat != y):
                    count+=1
                    self.w[y, :] = self.w[y, :] + local_rate*x 
                    self.w[y_hat, :] = self.w[y_hat, :] - local_rate*x 
                    self.bias[y] = self.bias[y] + local_rate
                    self.bias[y_hat] = self.bias[y_hat] - local_rate
   
    def predict_point(self, to_predict):
            normalize_to_predict = self.normalize_vector(to_predict, np.average(self.train_x), np.var(self.train_x))
            return np.argmax(np.dot(self.w, normalize_to_predict) + self.bias)

    def predict(self, predict_set):
        predicted = []
        for to_predict in predict_set:
            predicted.append(self.predict_point(to_predict))
        return predicted

    def shuffle(self, array_x, array_y):
        temp = np.random.get_state()
        np.random.shuffle(array_x)
        np.random.set_state(temp)
        np.random.shuffle(array_y)


    def normalize_vector(self, vector, avg, var):
        normalize_vector = []
        for x in vector:
            normalize_vector.append((x - avg)/(var))
        return np.asarray(vector)

    def normalize_data_set(self, data_set):
        avg = np.average(data_set)
        var = np.var(data_set)
        normalized_data_set = []
        for x in data_set:
            normalized_data_set.append(self.normalize_vector(x, avg, var))
        return normalized_data_set

def parse_file(path):
    c = []
    f = open(path)
    content = f.read()
    f.close()
    lines = content.split('\n')
    for line in lines:
       c.append(np.asarray(change_tuple(line.split(','))))
    return c


def change_tuple(array):
    new_array = []
    for x in array:
       new_array.append(float(x))
    return new_array

def main():
    train_x = parse_file('train_x.txt')
    train_y = parse_file('train_y.txt')
    learn_x = train_x[:int(0.8*240)]
    learn_y = train_y[:int(0.8*240)]
    test_x = train_x[int(0.8 * 240):]
    test_y = train_y[int(0.8 * 240):]

    k = preceptron(learn_x, learn_y)
    k.train()
    
    # k.train()
    # print()
    # result = (k.predict(test_x))
    # print(get_success_rate(result, train_y))
    
def get_success_rate(result, true_cla):
    count = 0
    for i in range(len(result)):
        if(result[i] == true_cla[i]):
            count+=1
    return count / len(result) * 100        

main()
