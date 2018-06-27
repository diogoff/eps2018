
import sys
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------

fname = 'tomo_kb5_reliable.hdf'
print 'Reading:', fname
f = h5py.File(fname, 'r')

X = []
Y = []

for pulse in f:
    g = f[pulse]
    kb5 = g['kb5'][:]/1e6
    tomo = g['tomo'][:]/1e6
    for i in range(tomo.shape[0]):
        X.append(tomo[i].flatten())
        Y.append(kb5[i])

f.close()

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

X = np.transpose(X)
Y = np.transpose(Y)

print 'X:', X.shape, X.dtype
print 'Y:', Y.shape, Y.dtype

# -------------------------------------------------------------------------

M = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float32)

print 'M:', M.shape, M.dtype

# -------------------------------------------------------------------------

import theano
import theano.tensor as T
from theano.printing import pydotprint

X = theano.shared(X, 'X')
Y = theano.shared(Y, 'Y')
M = theano.shared(M, 'M')

loss = T.mean(T.abs_(T.dot(M, Y) - X))

grad = T.grad(loss, M)

# -------------------------------------------------------------------------

lr = 0.01
momentum = 0.999

m = theano.shared(M.get_value() * np.float32(0.))
v = momentum * m - lr * grad

updates = []
updates.append((m, v))
updates.append((M, M + momentum * v - lr * grad))

# -------------------------------------------------------------------------

train = theano.function(inputs=[], outputs=[loss], updates=updates)

pydotprint(train, outfile='train.png', compact=False)  

epochs = 1000000

fname = 'train.log'
print 'Writing:', fname
f = open(fname, 'w')

print '%-10s %10s %20s' % ('time', 'epoch', 'loss')

try:
    t0 = time.time()
    for epoch in range(epochs):
        outputs = train()
        loss_value = outputs[0]
        t = time.strftime('%H:%M:%S', time.gmtime(time.time()-t0))
        print '\r%-10s %10d %20.12f' % (t, epoch+1, loss_value),
        sys.stdout.flush()
        f.write('%-10s %10d %20.12f\n' % (t, epoch+1, loss_value))
except KeyboardInterrupt:
    pass
print

f.close()

# -------------------------------------------------------------------------

M = M.get_value()

print 'M:', M.shape, M.dtype

fname = 'M.npy'
print 'Writing:', fname
np.save(fname, M)
