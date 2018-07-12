import numpy as np

myarray = np.array([1,2,3,4,5,6])

print(2 * myarray)
# add 10 to each element
print(myarray + 10)
# square each element
print(myarray ** 2)
# multiply two arrays element by element
print(myarray * myarray)
# is each element less than 4?
print(myarray < 4)
# is each element even?
print(myarray % 2 == 0)

print(len(myarray))

# works for numbers
print(round(3.1415926, 2))
# does not work for array
print(round(np.array([1.23456, 2.34567]), 2))

# correct way to do round
print(np.array([1.23456, 2.34567]).round(2))
