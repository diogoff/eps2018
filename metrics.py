
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim, compare_psnr, compare_nrmse

# -------------------------------------------------------------------------

fname = 'tomo_kb5_reliable.hdf'
print 'Reading:', fname
f = h5py.File(fname, 'r')

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

f.close()

X_valid = np.array(X_valid, dtype=np.float32)
Y_valid = np.array(Y_valid, dtype=np.float32)

X_valid = np.transpose(X_valid)
Y_valid = np.transpose(Y_valid)

print 'X_valid:', X_valid.shape, X_valid.dtype
print 'Y_valid:', Y_valid.shape, Y_valid.dtype

# -------------------------------------------------------------------------

fname = 'M.npy'
print 'Writing:', fname
M = np.load(fname)

print 'M:', M.shape, M.dtype

# -------------------------------------------------------------------------

X_pred = np.dot(M, Y_valid)

print 'X_pred:', X_pred.shape, X_pred.dtype

loss = np.mean(np.abs(X_pred - X_valid))

print 'loss:', loss

# -------------------------------------------------------------------------

list_ssim = []
list_pnsr = []
list_nrmse = []

print '%2s: %10s %10s %10s' % ('j', 'ssim', 'pnsr', 'nrmse')

for j in range(X_pred.shape[1]):
    img_0 = np.clip(X_valid[:,j].reshape((196, 115)), 0., 1.)
    img_1 = np.clip(X_pred[:,j].reshape((196, 115)), 0., 1.)

    ssim = compare_ssim(img_0, img_1)
    pnsr = compare_psnr(img_0, img_1)
    nrmse = compare_nrmse(img_0, img_1, norm_type='min-max')
    
    print '%2d: %10.6f %10.6f %10.6f' % (j, ssim, pnsr, nrmse)
    
    list_ssim.append(ssim)
    list_pnsr.append(pnsr)
    list_nrmse.append(nrmse)
    
    #fig, ax = plt.subplots(ncols=2)
    #ax[0].imshow(img_0, vmin=0., vmax=1.)
    #ax[1].imshow(img_1, vmin=0., vmax=1.)
    #plt.show()

print 'avg. ssim: %10.6f' % (np.mean(list_ssim))
print 'avg. pnsr: %10.6f' % (np.mean(list_pnsr))
print 'avg. nrmse: %10.6f' % (np.mean(list_nrmse))
