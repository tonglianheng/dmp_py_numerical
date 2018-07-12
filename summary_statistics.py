import numpy as np

mylist = np.array([0.94, 0.60, 0.74, 0.21, 0.50])

# min and max
print('min =', np.amin(mylist))
print('max =', np.amax(mylist))

# averages, sum
print('sum =', np.sum(mylist))
print('mean =', np.mean(mylist))
print('median =', np.median(mylist))

# variance and standard deviation
print('var =', np.var(mylist))
print('std =', np.std(mylist))

# correlation coeffients between two arrays
otherlist = np.array([0.14, 0.45, 0.57, 0.87, 0.29])
print('corr =\n', np.corrcoef(mylist, otherlist))

# build a 5 by 2 matrix using mylist and otherlist
mat = np.matrix([mylist, otherlist])

print('mat =\n', mat)
print('mean over rows =\n', np.mean(mat, 0))
print('mean over cols =\n', np.mean(mat, 1))
print('corr coeff between rows in mat =\n',
      np.corrcoef(mat, rowvar=True))
print('corr coeff between cols in mat =\n',
      np.corrcoef(mat, rowvar=False))
