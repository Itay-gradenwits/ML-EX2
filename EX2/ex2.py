import numpy as np
from numpy.lib.function_base import select
from numpy.linalg.linalg import cond
import random

from numpy.random import shuffle


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

def get_success_rate(result, true_cla):
    count = 0
    for i in range(len(result)):
        if(result[i] == true_cla[i]):
            count+=1
    return count / len(result) * 100        

class KnnModel:

    def __init__(self, train_x, train_y) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.k = 5

    def classify(self, test_x, k: int):
        classification = []
        for x in test_x:
            classification.append(self.classify_x(x, k))
        return classification

    def classify_x(self, x, k: int):
        closest_points = self.get_closest_points(x,k)
        chosen_value = self.get_most_decisions(closest_points)
        return chosen_value
    
    def get_closest_points(self, x , k: int):
        l = list(np.copy(self.train_x))
        l.sort(key=lambda p:np.linalg.norm(p-x,0))
        k_closest = l[:min(len(l),k)]
        return k_closest

    def get_most_decisions(self, k_closest):
        classifications = []
        for close_point in k_closest:
            classifications.append(self.train_y[self.find_index(close_point, self.train_x)][0])
        return max(set(classifications), key = classifications.count)
        
    def find_index(self, point, array):
        for i in range (len(array)):
            if ((array[i] == point).all()):
                return i
        return -1

    def predict(self, predict_x):
        return self.classify(predict_x, self.k)

class preceptron:
    def __init__(self, train_x, train_y) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.w = np.zeros((3,5))
        self.bias = np.zeros(3)
        self.rate = 0.5
    
    def train(self):
        local_rate = self.rate
        train_x_local = self.normalize_data_set(self.train_x, np.average(self.train_x), np.average(np.multiply(self.train_x, self.train_x)))
        train_y_local = np.copy(self.train_y)
        epochs = 50
        for e in range(epochs):
            self.shuffle(train_x_local, train_y_local)
            for x, y_iter in zip(train_x_local, train_y_local):
                y_hat = np.argmax(np.dot(self.w,x))
                y = int(y_iter[0])
                if(y_hat != y):
                    self.w[y, :] = self.w[y, :] + local_rate*x 
                    self.w[y_hat, :] = self.w[y_hat, :] - local_rate*x 
                    self.bias[y] = self.bias[y] + local_rate
                    self.bias[y_hat] = self.bias[y_hat] - local_rate
   
    def predict_point(self, to_predict):
            return np.argmax(np.dot(self.w, to_predict) + self.bias)

    def predict(self, predict_set):
        avg = np.average(self.train_x)
        avg_2 = np.average(np.multiply(self.train_x, self.train_x))
        predicted = []
        normalized_predict_set = self.normalize_data_set(predict_set, avg, avg_2)
        for to_predict in normalized_predict_set:
            print(to_predict)
            predicted.append(self.predict_point(to_predict))
        return predicted

    def shuffle(self, array_x, array_y):
        temp = np.random.get_state()
        np.random.shuffle(array_x)
        np.random.set_state(temp)
        np.random.shuffle(array_y)

    def normalize_data_set(self, data_set, avg, avg_powerd_by_2):
        normalized_data_set = []
        for vector in data_set:
            normalized_data_set.append(np.divide(np.subtract(vector,avg),avg_powerd_by_2 - avg* avg))
        return normalized_data_set 

class SvmModel:
    def __init__(self, train_x, train_y) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.w = np.zeros((3,5))
        self.bias = np.zeros(3)
        self.rate = 0.5
    
    def train(self):
        local_rate = self.rate
        train_x_local = self.normalize_data_set(self.train_x, np.average(self.train_x), np.average(np.multiply(self.train_x, self.train_x)))
        train_y_local = np.copy(self.train_y)
        epochs = 300
        for e in range(epochs):
            self.shuffle(train_x_local, train_y_local)
            for x, y_iter in zip(train_x_local, train_y_local):
                y_hat = np.argmax(np.dot(self.w,x))
                y = int(y_iter[0])
                if(y_hat != y):
                    self.w[y, :] = self.w[y, :] + local_rate*x 
                    self.w[y_hat, :] = self.w[y_hat, :] - local_rate*x 
                    self.bias[y] = self.bias[y] + local_rate
                    self.bias[y_hat] = self.bias[y_hat] - local_rate
   
    def predict_point(self, to_predict):
            return np.argmax(np.dot(self.w, to_predict) + self.bias)

    def predict(self, predict_set):
        avg = np.average(self.train_x)
        avg_2 = np.average(np.multiply(self.train_x, self.train_x))
        predicted = []
        normalized_predict_set = self.normalize_data_set(predict_set, avg, avg_2)
        for to_predict in normalized_predict_set:
            predicted.append(self.predict_point(to_predict))
        return predicted

    def shuffle(self, array_x, array_y):
        temp = np.random.get_state()
        np.random.shuffle(array_x)
        np.random.set_state(temp)
        np.random.shuffle(array_y)

    def normalize_data_set(self, data_set, avg, avg_powerd_by_2):
        normalized_data_set = []
        for vector in data_set:
            normalized_data_set.append(np.divide(np.subtract(vector,avg),avg_powerd_by_2 - avg* avg))
        return normalized_data_set 


def main():
    train_x = parse_file('train_x.txt')
    train_y = parse_file('train_y.txt')

    learn_x = train_x[:int(0.8*240)]
    learn_y = train_y[:int(0.8*240)]
    test_x = train_x[int(0.8 * 240):]
    test_y = train_y[int(0.8 * 240):]


    model = preceptron(learn_x, learn_y)
    model.train()
    cla = model.predict(test_x)
    print(get_success_rate(cla, test_y))
    # model.train()
    # print(model.predict(to_predict_x))


main()