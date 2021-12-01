import numpy as np
from numpy.lib.function_base import select
from numpy.linalg.linalg import cond
import sys
from numpy.random import shuffle


def write_results_to_output(result_knn, result_perceptron, result_svm, result_pa, output_path):
    f = open(output_path, 'w')
    for i in range(len(result_pa)):
        f.write(f"knn: {result_knn[i]}, perceptron: {result_perceptron[i]}, svm: {result_svm[i]}, pa: {result_pa[i]}\n")
    f.close()

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

def loss(x, y, w, b):
    loss = 0
    for i in range(len(w)):
        if(i != y):
            arg2 = np.float_(1 - np.dot(w[y], x) - np.dot(w[i], x) - b[y] - b[i])
            loss = max(loss, arg2)
    return loss


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
        return int(max(set(classifications), key = classifications.count))
        
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
        train_x_local = self.normalize_data_set(self.train_x, np.average(self.train_x, axis=0), np.var(self.train_x, axis=0))
        train_y_local = np.copy(self.train_y)
        epochs = 300
        for e in range(1,epochs):
            self.shuffle(train_x_local, train_y_local)
            for x, y_iter in zip(train_x_local, train_y_local):
                y_hat = np.argmax(np.dot(self.w,x) + self.bias)
                y = int(y_iter[0])
                if(y_hat != y):
                    self.w[y, :] = self.w[y, :] + local_rate*x 
                    self.w[y_hat, :] = self.w[y_hat, :] - local_rate*x 
                    self.bias[y] = self.bias[y] + local_rate
                    self.bias[y_hat] = self.bias[y_hat] - local_rate
            self.rate = self.rate / np.sqrt(e)
   
    def predict_point(self, to_predict):
            return np.argmax(np.dot(self.w, to_predict) + self.bias)

    def predict(self, predict_set):
        avg = np.average(self.train_x, axis=0)
        var = np.var(self.train_x, axis=0)
        predicted = []
        normalized_predict_set = self.normalize_data_set(predict_set, avg, var)
        for to_predict in normalized_predict_set:
            predicted.append(self.predict_point(to_predict))
        return predicted

    def shuffle(self, array_x, array_y):
        temp = np.random.get_state()
        np.random.shuffle(array_x)
        np.random.set_state(temp)
        np.random.shuffle(array_y)

    def normalize_data_set(self, data_set, avg, var):
        normalized_data_set = []
        for vector in data_set:
            normalized_data_set.append(np.divide(np.subtract(vector,avg),var))
        return normalized_data_set 

class SvmModel:
    def __init__(self, train_x, train_y) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.w = np.zeros((3,5))
        self.bias = np.zeros(3)
        self.eta = 0.1
        self.lam = 0.01

    
    def train(self):
        local_eta = self.eta
        local_lam = self.lam
        train_x_local = self.normalize_data_set(self.train_x, np.average(self.train_x, axis=0), np.var(self.train_x, axis=0))
        train_y_local = np.copy(self.train_y)
        epochs = 300
        for e in range(1,epochs):
            self.shuffle(train_x_local, train_y_local)
            for x, y_iter in zip(train_x_local, train_y_local):
                y = int(y_iter[0])
                new_w = self.get_w_without_y(self.w, y)
                new_bias = self.get_bias_without_y(self.bias, y)
                y_hat = np.argmax(np.dot(new_w,x) + new_bias)
                
                if(loss(x, y, self.w, self.bias) > 0):
                    self.w[y, :] = self.w[y, :] * (1 - local_eta * local_lam) + local_eta * x
                    self.w[y_hat, :] = self.w[y_hat, :] * (1 - local_eta * local_lam) - local_eta * x
                    self.bias[y] = self.bias[y] * (1 - local_eta * local_lam) + local_eta
                    self.bias[y_hat] = self.bias[y_hat] * (1 - local_eta * local_lam) - local_eta
                    
                self.update_without_y_y_hat(self.w, self.bias, y, y_hat, local_eta, local_lam)

    def get_w_without_y(self, cur_w, y):
        new_w = []
        for i in range(len(cur_w)):
            if(i == y):
                continue
            new_w.append(cur_w[i])
        return new_w
    
    def get_bias_without_y(self, cur_bias, y):
        new_bias = []
        for i in range(len(cur_bias)):
            if( i == y):
                continue
            new_bias.append(cur_bias[i])
        return new_bias

    def update_without_y_y_hat(self, w, b, y, y_hat, local_eta, local_lam):
        for i in range(len(w)):
            if(i == y or i == y_hat):
                continue
            w[i, :] = w[i, :] * (1-local_lam * local_eta)
            b[i] = b[i] * (1 - local_eta * local_lam)


    def predict_point(self, to_predict):
            return np.argmax(np.dot(self.w, to_predict) + self.bias)

    def predict(self, predict_set):
        avg = np.average(self.train_x, axis=0)
        var = np.var(self.train_x, axis=0)
        predicted = []
        normalized_predict_set = self.normalize_data_set(predict_set, avg, var)
        for to_predict in normalized_predict_set:
            predicted.append(self.predict_point(to_predict))
        return predicted

    def shuffle(self, array_x, array_y):
        temp = np.random.get_state()
        np.random.shuffle(array_x)
        np.random.set_state(temp)
        np.random.shuffle(array_y)

    def normalize_data_set(self, data_set, avg, var):
        normalized_data_set = []
        for vector in data_set:
            normalized_data_set.append(np.divide(np.subtract(vector,avg),var))
        return normalized_data_set 


class PA:
    def __init__(self, train_x, train_y) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.w = np.zeros((3,5))
        self.bias = np.zeros(3)
    
    def train(self):
        train_x_local = self.normalize_data_set(self.train_x, np.average(self.train_x, axis=0), np.var(self.train_x, axis=0))
        train_y_local = np.copy(self.train_y)
        epochs = 500
        for e in range(1,epochs):
            self.shuffle(train_x_local, train_y_local)
            for x, y_iter in zip(train_x_local, train_y_local):
                y_hat = np.argmax(np.dot(self.w,x) + self.bias)
                y = int(y_iter[0])
                if(y_hat != y):
                    l = loss(x,y,self.w, self.bias) / (2 * (np.linalg.norm(x) ** 2))
                    self.w[y, :] = self.w[y, :] + l*x 
                    self.w[y_hat, :] = self.w[y_hat, :] - l*x 
                    self.bias[y] = self.bias[y] + l
                    self.bias[y_hat] = self.bias[y_hat] - l
   
    def predict_point(self, to_predict):
            return np.argmax(np.dot(self.w, to_predict) + self.bias)

    def predict(self, predict_set):
        avg = np.average(self.train_x, axis=0)
        var = np.var(self.train_x, axis=0)
        predicted = []
        normalized_predict_set = self.normalize_data_set(predict_set, avg, var)
        for to_predict in normalized_predict_set:
            predicted.append(self.predict_point(to_predict))
        return predicted

    def shuffle(self, array_x, array_y):
        temp = np.random.get_state()
        np.random.shuffle(array_x)
        np.random.set_state(temp)
        np.random.shuffle(array_y)

    def normalize_data_set(self, data_set, avg, var):
        normalized_data_set = []
        for vector in data_set:
            normalized_data_set.append(np.divide(np.subtract(vector,avg),var))
        return normalized_data_set 



def main2():
    train_x = parse_file('train_x.txt')
    train_y = parse_file('train_y.txt')

    learn_x = train_x[:int(0.8*240)]
    learn_y = train_y[:int(0.8*240)]
    test_x = train_x[int(0.8 * 240):]
    test_y = train_y[int(0.8 * 240):]


    model = SvmModel(learn_x, learn_y)
    model.train()
    cla = model.predict(test_x)
    print(get_success_rate(cla, test_y))
    # model.train()
    # print(model.predict(to_predict_x))

def main():
    train_x = parse_file(sys.argv[1])
    train_y = parse_file(sys.argv[2])
    test_x = parse_file(sys.argv[3])
    output_path = sys.argv[4]
    knn_model = KnnModel(train_x, train_y)
    perceptron_model = preceptron(train_x, train_y)
    svm_model = SvmModel(train_x, train_y)
    pa_model = PA(train_x, train_y)

    perceptron_model.train()
    svm_model.train()
    pa_model.train()


    result_knn = knn_model.predict(test_x)
    result_perceptron = perceptron_model.predict(test_x)
    result_svm = svm_model.predict(test_x)
    result_pa = pa_model.predict(test_x)

    write_results_to_output(result_knn, result_perceptron, result_svm, result_pa, output_path)




main()