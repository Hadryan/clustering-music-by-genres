import random

groups = 13
div = [3, 4, 3, 3]

# Jazz, Classical, EDM, Rock

conn = [[15, 2, 2, 1, 0],
        [3, 3, 2, 2, 0],
        [11, 2, 0, 2, 0],
        [5, 9, 1, 0, 1],
        [0, 9, 0, 2, 1],
        [2, 3, 2, 1, 1],
        [6, 9, 0, 1, 1],
        [5, 0, 12, 6, 2],
        [0, 0, 5, 4, 2],
        [0, 3, 7, 2, 2],
        [3, 1, 10, 15, 3],
        [0, 2, 4, 9, 3],
        [1, 1, 6, 12, 3]]


def off(key):
    x = conn[key][-1]
    return sum(conn[key][:x])


id = 0
print '{"nodes":[{"id": -1 ,"group": 5 },'
for key, val in enumerate(conn):
    for keyx, valx in enumerate(val[:-1]):
        for i in range(valx):
            print '{"id":', str(id), ',"group":', keyx, '},'
            id += 1

id = 0
print '],"links":['
for key, val in enumerate(conn):
    target = id
    for keyx, valx in enumerate(val[:-1]):
        for i in range(valx):
            targetx = -1 if target + off(key) == id else target + off(key)
            wt = 0.1 if target + off(key) == id else random.random()
            print '{"source":', str(id), ',"target":', targetx, ',"value":', wt, '},'
            id += 1
print ']}'
