
import sys
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------

fname = 'tomo_kb5_reliable.hdf'
print 'Reading:', fname
f = h5py.File(fname, 'r')

X_train = []
Y_train = []

X_valid = []
Y_valid = []

k = 0
for pulse in f:
    g = f[pulse]
    kb5 = g['kb5'][:]/1e6
    tomo = g['tomo'][:]/1e6
    for i in range(tomo.shape[0]):
        k += 1
        if k % 10 == 0:
            X_valid.append(tomo[i].flatten())
            Y_valid.append(kb5[i])
        else:
            X_train.append(tomo[i].flatten())
            Y_train.append(kb5[i])

f.close()

X_train = np.array(X_train, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.float32)

X_valid = np.array(X_valid, dtype=np.float32)
Y_valid = np.array(Y_valid, dtype=np.float32)

X_train = np.transpose(X_train)
Y_train = np.transpose(Y_train)

X_valid = np.transpose(X_valid)
Y_valid = np.transpose(Y_valid)

print 'X_train:', X_train.shape, X_train.dtype
print 'Y_train:', Y_train.shape, Y_train.dtype

print 'X_valid:', X_valid.shape, X_valid.dtype
print 'Y_valid:', Y_valid.shape, Y_valid.dtype

# -------------------------------------------------------------------------

M = np.zeros((X_train.shape[0], Y_train.shape[0]), dtype=np.float32)

print 'M:', M.shape, M.dtype

# -------------------------------------------------------------------------

import theano
import theano.tensor as T
from theano.printing import pydotprint

X_train = theano.shared(X_train, 'X_train')
Y_train = theano.shared(Y_train, 'Y_train')

X_valid = theano.shared(X_valid, 'X_valid')
Y_valid = theano.shared(Y_valid, 'Y_valid')

M = theano.shared(M, 'M')

loss_train = T.mean(T.abs_(T.dot(M, Y_train) - X_train))
loss_valid = T.mean(T.abs_(T.dot(M, Y_valid) - X_valid))

grad = T.grad(loss_train, M)

# -------------------------------------------------------------------------

lr = 0.01
momentum = 0.999

m = theano.shared(M.get_value() * np.float32(0.))
v = momentum * m - lr * grad

updates = []
updates.append((m, v))
updates.append((M, M + momentum * v - lr * grad))

# -------------------------------------------------------------------------

train = theano.function(inputs=[], outputs=[loss_train], updates=updates)
valid = theano.function(inputs=[], outputs=[loss_valid], updates=[])

pydotprint(train, outfile='train.png', compact=False)  
pydotprint(valid, outfile='valid.png', compact=False)  

epochs = 1000000

fname = 'train.log'
print 'Writing:', fname
f = open(fname, 'w')

print '%-10s %10s %20s %20s' % ('time', 'epoch', 'loss', 'val_loss')

try:
    t0 = time.time()
    for epoch in range(epochs):
        loss_value_train = train()[0]
        loss_value_valid = valid()[0]
        t = time.strftime('%H:%M:%S', time.gmtime(time.time()-t0))
        print '\r%-10s %10d %20.12f %20.12f' % (t, epoch+1, loss_value_train, loss_value_valid),
        sys.stdout.flush()
        f.write('%-10s %10d %20.12f %20.12f\n' % (t, epoch+1, loss_value_train, loss_value_valid))
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
