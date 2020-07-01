import onnx
from onnx import numpy_helper
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt

cntr_address = "/home/amaleki/Downloads/ARCH-COMP2020/benchmarks/Benchmark9-Tora/controllerTora.onnx"
onnx_model = onnx.load(cntr_address)
inits = onnx_model.graph.initializer
w1 = numpy_helper.to_array(inits[1]).reshape(100, 4).T
b1 = numpy_helper.to_array(inits[2])
w2 = numpy_helper.to_array(inits[3]).reshape(100, 100).T
b2 = numpy_helper.to_array(inits[4])
w3 = numpy_helper.to_array(inits[5]).reshape(100, 100).T
b3 = numpy_helper.to_array(inits[6])
w4 = numpy_helper.to_array(inits[7]).reshape(1, 100).T
b4 = numpy_helper.to_array(inits[8])
w5 = np.array([[1]], dtype=np.float32)
b5 = np.array([-10.], dtype=np.float32)

W = [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5]

keras_model = Sequential()
keras_model.add(Dense(100, activation="relu", input_shape=(4,)))
keras_model.add(Dense(100, activation="relu"))
keras_model.add(Dense(100, activation="relu"))
keras_model.add(Dense(1, activation="relu"))
keras_model.add(Dense(1, activation="linear"))

keras_model.set_weights(W)

def generate_data(nsize):
    lims = (-1.5, 1.5)
    X = np.random.random((nsize, 4))
    X *= 3
    X -= 1.5
#     x1 = np.linspace(x1lim[0], x1lim[1], 200)
#     x2 = np.linspace(x2lim[0], x2lim[1], 200)
#     x3 = np.linspace(x3lim[0], x3lim[1], 200)
#     x4 = np.linspace(x4lim[0], x4lim[1], 200)
#     X = np.concatenate((
#                        np.random.choice(x1, size=(nsize, 1)), 
#                        np.random.choice(x2, size=(nsize, 1)), 
#                        np.random.choice(x3, size=(nsize, 1)),
#                        np.random.choice(x4, size=(nsize, 1))
#                        ), axis=1)
    
    y = keras_model.predict(X)
    return X, y


X_train, y_train = generate_data(1000000)
X_test, y_test = generate_data(1000)

keras_smaller_model = Sequential()
keras_smaller_model.add(Dense(50, activation="relu", input_shape=(4,)))
keras_smaller_model.add(Dense(50, activation="relu"))
keras_smaller_model.add(Dense(50, activation="relu"))
keras_smaller_model.add(Dense(1, activation="relu"))
keras_smaller_model.add(Dense(1, activation="linear"))

keras_smaller_model.compile(optimizer="adam", loss='mse')
history = keras_smaller_model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=10, verbose=1) #, callbacks=[callback]

plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


keras_smallest_model = Sequential()
keras_smallest_model.add(Dense(25, activation="relu", input_shape=(4,)))
keras_smallest_model.add(Dense(25, activation="relu"))
keras_smallest_model.add(Dense(25, activation="relu"))
keras_smallest_model.add(Dense(1, activation="relu"))
keras_smallest_model.add(Dense(1, activation="linear"))
keras_smallest_model.compile(optimizer="adam", loss='mse')
history = keras_smallest_model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=10, verbose=1) #, callbacks=[callback]

plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

x = np.array([1,0,1.2,-0.4]).reshape(1,4)
v1 = keras_model.predict(x)
v2 = keras_smaller_model.predict(x)
v3 = keras_smallest_model.predict(x)
print(v1, v2, v3)

X_test, y_test = generate_data(1000)
y1 = keras_model.predict(X_test)
y2 = keras_smaller_model.predict(X_test)
y3 = keras_smallest_model.predict(X_test)

plt.figure(figsize=(12,8))
plt.scatter(y_test, y2, color='r', marker='.')
plt.scatter(y_test, y3, color='g', marker='.')
plt.plot(y_test, y_test)
plt.show()

plt.figure(figsize=(12,8))
plt.scatter(y_test, y2-y_test, color='r', marker='.')
plt.scatter(y_test, y3-y_test, color='g', marker='.')
plt.show()

def keras_to_nnet(model, out_file):
    w = model.get_weights()
    l = model.layers
    with open(out_file, "w") as f:
        n_layers = len(l)
        f.write("%d \n"%n_layers)
        for i in range(n_layers):
            f.write("%d, "%w[i*2].shape[0])
        f.write("%d \n"%w[-1].shape[0])
        for i in range(5):
            f.write("0\n")
        for k in range(0, len(w), 2):
            n_rows, n_cols = w[k].shape
            for i in range(n_cols):
                for j in range(n_rows-1):
                    f.write("%0.5f, " %w[k][j, i])
                f.write("%0.5f \n" %w[k][n_rows-1, i])
            for i in range(n_cols):
                f.write("%0.5f \n"%w[k+1][i])

keras_to_nnet(keras_smaller_model, "controllerTora_smaller.nnet")
keras_to_nnet(keras_smallest_model, "controllerTora_smallest.nnet")
