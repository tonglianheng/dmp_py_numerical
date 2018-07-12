import numpy as np
heights = np.array([175, 177, 172, 180])
print(heights[0], heights[2], heights[-1])

first_three = heights[0:3]
print(first_three)

second_fourth = heights[1::2]
print(second_fourth)
first_two = heights[:2]
print(first_two)

reverse = heights[-1::-1]
print(reverse)

# no. daily messages sent between people
alice, bob, charlie = 0, 1, 2
messages = np.array(
  [[ 0,  3, 17],
   [12,  0, 11],
   [ 9,  5,  0]]
)

n_sent = messages[alice, bob]
print(f'Alice sent {n_sent} msgs to Bob')

comm_first_two = messages[0:2, 0:2]
print(comm_first_two)
