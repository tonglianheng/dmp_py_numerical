import numpy as np

mylist = [1,2,3,4,5]
myarray = np.array(mylist)
print(type(myarray))
print(myarray)

myarray = np.array([1, 2, '3'])
print(myarray)

# 1D array of zeros
initial_vec = np.zeros(10)
print(initial_vec)

sequence = np.arange(10)
print(sequence)
sequence = np.arange(10,20,2)
print(sequence)
