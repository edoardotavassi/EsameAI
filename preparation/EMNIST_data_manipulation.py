from matplotlib.pyplot import axis
import numpy as np
from emnist import extract_training_samples, extract_test_samples
from numpy import save

x_train, y_train = extract_training_samples("balanced")
x_test, y_test = extract_test_samples("balanced")
print("Original Train Shapes: ")
print("Data: ", x_train.shape)
print("Labels: ", y_train.shape)
print("Original Test Shapes")
print("Data: ", y_test.shape)
print("Labels: ", x_test.shape)

arr = []
count = 0
for i in range(y_train.shape[0]):
    if y_train[i] == 1:
        pass
    elif y_train[i] == 2:
        pass
    elif y_train[i] == 3:
        pass
    elif y_train[i] == 4:
        pass
    elif y_train[i] == 28:
        pass
    elif y_train[i] == 29:
        pass
    elif y_train[i] == 33:
        pass
    else:
        arr.append(i)
        count += 1

print("\nNumber of elements to delete[Train] %s" % count)
print("Deletin train data")
x_train = np.delete(x_train, arr, axis=0)
print("Deleting train labeles")
y_train = np.delete(y_train, arr, axis=0)

arr2 = []
count = 0
for i in range(y_test.shape[0]):
    if y_test[i] == 1:
        pass
    elif y_test[i] == 2:
        pass
    elif y_test[i] == 3:
        pass
    elif y_test[i] == 4:
        pass
    elif y_test[i] == 28:
        pass
    elif y_test[i] == 29:
        pass
    elif y_test[i] == 33:
        pass
    else:
        arr2.append(i)
        count += 1

print("\nNumber of elements to delete[test] %s" % count)
print("Deletin test data")
x_test = np.delete(x_test, arr2, axis=0)
print("Deleting test labeles")
y_test = np.delete(y_test, arr2, axis=0)

print("\nAdapting labels:")
for i in range(y_test.shape[0]):
    if y_test[i] == 1:
        y_test[i] = 0
    elif y_test[i] == 2:
        y_test[i] = 1
    elif y_test[i] == 3:
        y_test[i] = 2
    elif y_test[i] == 4:
        y_test[i] = 3
    elif y_test[i] == 28:
        y_test[i] = 4
    elif y_test[i] == 29:
        y_test[i] = 5
    elif y_test[i] == 33:
        y_test[i] = 6
    else:
        print("out bound: %s" % y_test[i])
        

for i in range(y_train.shape[0]):
    if y_train[i] == 1:
        y_train[i] = 0
    elif y_train[i] == 2:
        y_train[i] = 1
    elif y_train[i] == 3:
        y_train[i] = 2
    elif y_train[i] == 4:
        y_train[i] = 3
    elif y_train[i] == 28:
        y_train[i] = 4
    elif y_train[i] == 29:
        y_train[i] = 5
    elif y_train[i] == 33:
        y_train[i] = 6
    else:
        print("out bound: %s" % y_train[i])


print("\nFinal Train Shapes: ")
print("Data: ", x_train.shape)
print("Labels: ", y_train.shape)
print("Final Test Shapes")
print("Data: ", y_test.shape)
print("Labels: ", x_test.shape)

# save to npy file
print("\nSaving files.")
save("../datasets/train_data.npy", x_train)
save("../datasets/train_labels.npy", y_train)
save("../datasets/test_data.npy", x_test)
save("../datasets/test_labels.npy", y_test)
print("Files saved.")