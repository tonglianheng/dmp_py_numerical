import numpy as np

mat1 = np.array([[1,2],
                 [3,4]])
print(mat1 * mat1)

mat2 = np.matrix([[1,2],
                  [3,4]])
print(mat2 * mat2)

print(mat1 @ mat1)
print(mat2 @ mat2)
