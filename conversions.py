import numpy as np

mylist = np.arange(12)

# we can reshape list into a matrix
mymat = mylist.reshape(4, 3)

print(mylist)
print(mymat)

print(mylist)
myflist = mylist.astype(
  np.dtype('float')
)
print(myflist)

native_list = myflist.tolist()
print(native_list)
print(type(native_list))
