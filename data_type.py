import numpy as np
array = np.array([1, 2])
print(array.dtype)
array = np.array([1.0, 2.0])
print(array.dtype)

array = np.array([1, 2], dtype='float')
print(array, array.dtype)
array = np.array(['1', '2'], dtype='float')
print(array, array.dtype)

array = np.array([1, 2, 3])
new = array.astype('float')
print(new, new.dtype)

array = np.array([1,2], dtype='int64')
print(array, array.dtype)
array.dtype = 'int32'
print(array, array.dtype)

array = np.array([1,2], dtype='int64')
new = array.astype('int32')
print(new, new.dtype)
