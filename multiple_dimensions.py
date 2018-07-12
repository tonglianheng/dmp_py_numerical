import numpy as np
twodim = np.array(
  [[1, 2],
   [3, 4]]
)
threedim = np.array(
  [
    [[1, 2],
     [3, 4]],
    [[5, 6],
     [7, 8]]
  ]
)
print(twodim)
print('\n------------\n')
print(threedim)

print(twodim.shape)
print(threedim.shape)

twodim.shape = (4,1)
print(twodim)
twodim.shape = (1,4)
print(twodim)

# leads to an error
twodim.shape = (1,2)
