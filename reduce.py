import numpy as np
numbers = np.arange(1,11)
print(numbers)

total = np.add.reduce(numbers)
print(total)

minimum = np.minimum.reduce(numbers)
print(minimum)

# native python sum already exist
print(sum(numbers))

words = np.array(['ready', 'impossible',
                  'adaptable', 'able'])

def max_len(word1, word2):
    if len(word1) > len(word2):
        return word1
    else:
        return word2

v_max_len = np.frompyfunc(max_len, 2, 1)

longest = v_max_len.reduce(words)
print(longest)
