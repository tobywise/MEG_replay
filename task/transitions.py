import numpy as np
import random

# T MAZE
matrix = np.zeros((13, 13))

matrix[0, 1] = 1
matrix[[1, 1], [0, 2]] = 1
matrix[[2, 2], [1, 3]] = 1
matrix[[3, 3, 3, 3], [2, 4, 9, 10]] = 1
matrix[[4, 4], [3, 5]] = 1
matrix[[5, 5], [4, 6]] = 1
matrix[6, 5] = 1

matrix[7, 8] = 1
matrix[[8, 8], [7, 9]] = 1
matrix[[9, 9], [8, 3]] = 1
matrix[[10, 10], [3, 11]] = 1
matrix[[11, 11], [10, 12]] = 1
matrix[12, 11] = 1

matrix_keys = matrix.astype(int).astype(str)
keys = ['up', 'down', 'left', 'right']
random.shuffle(keys)

for i in range(matrix.shape[0]):
    one_idx = np.where(matrix[i, :] == 1)
    random.shuffle(keys)
    for j in range(len(one_idx[0])):
        matrix_keys[i, one_idx[0][j]] = keys[j]


## test
start = np.random.randint(0, 13)
previous_state = None

for i in range(0, 10):
    print previous_state
    print "Move {0}, start = {1}".format(i, start)
    allowed_moves = [i for n, i in enumerate(matrix_keys[start, :]) if not '0' in i and not n == previous_state]
    row = matrix_keys[start, :]
    response = raw_input("Enter move")
    next_state = np.where(row == response)
    previous_state = start
    start = next_state[0][0]

import seaborn as sns
sns.heatmap(matrix)

import random
random.seed(2)
a = range(10)
random.shuffle(a)
print a