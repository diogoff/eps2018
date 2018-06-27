
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------

from geom import *

kb5_los = get_kb5_los()

line = []

for (Rst, Zst, Rend, Zend) in kb5_los:
    xx = [Rst, Rend]
    yy = [Zst, Zend]
    line.append((xx, yy))
    
# -------------------------------------------------------------------------

fname = 'M.npy'
print 'Reading:', fname
M = np.load(fname)

print 'M:', M.shape, M.dtype

# -------------------------------------------------------------------------

nrows = 4
ncols = 14

fig, axes = plt.subplots(nrows, ncols)

ai = 0
aj = 0

for j in range(M.shape[1]):
    img = M[:,j].reshape((196, 115))
    print '%10.6f %10.6f %10.6f %10.6f %10.6f' % (np.min(img), np.max(img), np.mean(img), np.std(img), np.median(img))
    title = 'Line %d' % (j+1)
    vmax = np.max(img)
    num = 5
    fontsize = 'small'
    R0 = 1.708 - 2*0.02
    R1 = 3.988 + 3*0.02
    Z0 = -1.77 - 2*0.02
    Z1 = +2.13 + 2*0.02
    ax = axes[ai,aj]
    im = ax.imshow(img, cmap='plasma',
                   vmin=0., vmax=vmax,
                   extent=[R0, R1, Z0, Z1],
                   interpolation='none')
    (xx, yy) = line[j]
    ax.plot(xx, yy, 'r--')
    ax.axis('off')
    ax.set_title('Line %d' % (j+1), fontsize=fontsize)
    ticks = np.linspace(0., vmax, num=num)
    labels = ['%.4f' % tick for tick in ticks]
    #cb = fig.colorbar(im, ax=ax, ticks=ticks)
    #cb.ax.set_yticklabels(labels, fontsize=fontsize)
    #ax.tick_params(labelsize=fontsize)
    #ax.set_xlabel('R (m)', fontsize=fontsize)
    #ax.set_ylabel('Z (m)', fontsize=fontsize)
    #ax.set_xlim([R0, R1])
    #ax.set_ylim([Z0, Z1])
    #plt.setp(ax.spines.values(), linewidth=0.1)
    aj += 1
    if aj >= ncols:
        ai += 1
        aj = 0

fig.set_size_inches(17, 9)

plt.subplots_adjust(left=0.001, right=1.-0.001, bottom=0.001, top=1.-0.025, wspace=0.02, hspace=0.12)

plt.show()
