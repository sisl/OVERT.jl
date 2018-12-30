# more testing code
import numpy as np

# test: condense_list
W = []
b = []
for i in range(2):
	W.append(np.random.rand(2,2))
	b.append(np.random.rand(2,1))

x = np.array([[1.0],[1.0]])

# hand multiply:
y = x
for i in range(2):
	y = W[i]@y + b[i]

import parsing as p

Wc, bc = p.condense_list(W,b)
yc = Wc@x + bc

print("yc: ", yc)
print("y: ", y)
assert( all((yc-y) < 1e-6) )