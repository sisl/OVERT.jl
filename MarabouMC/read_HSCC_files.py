import numpy as np
from keras.models import save_model,  load_model, Sequential
from keras.layers import Dense

class HSCC_Controllers:
    def __init__(self, file_name):
        self.pointer = 0
        self.model = Sequential()
        self.layer_sizes = []
        self.weights_and_biasses = []
        with open(file_name, 'r') as f:
            self.file_data = f.readlines()
        self.get_sizes()
        self.get_weights()
        self.convert_to_keras()
        assert(len(self.file_data)==self.pointer)

    def read_n_lines(self, n=1, to_float=False, to_int=False):
        if n == 1:
            str = self.file_data[self.pointer].strip()
            if to_float: out = float(str)
            elif to_int: out = int(str)
            else: out = str
            self.pointer += 1
        else:
            out = []
            for i in range(n):
                str = self.file_data[self.pointer].strip()
                if to_float: out.append(float(str))
                elif to_int: out.append(int(str))
                else: return out.append(str)
                self.pointer += 1
        return out

    def get_sizes(self):
        self.layer_sizes.append(self.read_n_lines(to_int=True))
        output_size = self.read_n_lines(to_int=True) # get the output size
        n_hidden = self.read_n_lines(to_int=True) # get the number of hidden layers
        for i in range(n_hidden):
            self.layer_sizes.append(self.read_n_lines(to_int=True))
        self.layer_sizes.append(output_size)

    def get_weights(self):
        for n_layer in range(len(self.layer_sizes)-1):
            in_size, out_size = self.layer_sizes[n_layer], self.layer_sizes[n_layer+1]
            w = np.zeros((in_size, out_size))
            b = np.zeros((out_size, ))
            for n_output in range(out_size):
                w[:, n_output] = self.read_n_lines(in_size, to_float=True)
                b[n_output] = self.read_n_lines(to_float=True)

            self.weights_and_biasses.append(w)
            self.weights_and_biasses.append(b)
        self.weights_and_biasses = np.array(self.weights_and_biasses)

    def convert_to_keras(self):
        for i in range(len(self.layer_sizes)-1):
            if i == 0:
                self.model.add(Dense(self.layer_sizes[i + 1], activation="relu", input_dim=self.layer_sizes[i]))
            else:
                self.model.add(Dense(self.layer_sizes[i + 1], activation="relu"))
        self.model.set_weights(self.weights_and_biasses)

if __name__ == '__main__':
    #file_name = "/home/amaleki/Downloads/Neural-Network-Controller-Verification-Benchmarks-HSCC-2019-master/Benchmarks/Ex_2/modified_controller"
    file_name = "/home/amaleki/Downloads/Neural-Network-Controller-Verification-Benchmarks-HSCC-2019-master/Benchmarks/Ex_10/neural_network_controller_4"

    out_name = file_name +"_keras.h5"

    c = HSCC_Controllers(file_name)
    save_model(c.model, out_name)
    print(c.model.predict(np.array([2., -0.5, 3., -1]).reshape(1, -1)))