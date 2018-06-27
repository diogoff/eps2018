
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

from geom import *
from shapely.geometry.polygon import Polygon

# -------------------------------------------------------------------------

fname = 'bolo_kb5_reliable.hdf'
print 'Reading:', fname
f = h5py.File(fname, 'r')

pulse = '92213'
g = f[pulse]
kb5 = g['bolo'][:]/1e6
kb5_t = g['t'][:]

f.close()

kb5 = np.transpose(kb5)

print 'kb5:', kb5.shape, kb5.dtype
print 'kb5_t:', kb5_t.shape, kb5_t.dtype

# -------------------------------------------------------------------------

fname = 'M.npy'
print 'Reading:', fname
M = np.load(fname)

print 'M:', M.shape, M.dtype

# -------------------------------------------------------------------------

tomo = np.dot(M, kb5)
tomo = np.transpose(tomo)
tomo = np.reshape(tomo, (tomo.shape[0], 196, 115))

tomo_t = kb5_t

t = 47.0
i = np.argmin(np.fabs(tomo_t - t))
tomo = tomo[i:]
tomo_t = tomo_t[i:]

t = 54.5
i = np.argmin(np.fabs(tomo_t - t))
tomo = tomo[:i+1]
tomo_t = tomo_t[:i+1]

print 'tomo:', tomo.shape, tomo.dtype
print 'tomo_t:', tomo_t.shape, tomo_t.dtype

# -------------------------------------------------------------------------

vmax = 1.

fontsize = 'small'

R0 = 1.708 - 2*0.02
R1 = 3.988 + 3*0.02
Z0 = -1.77 - 2*0.02
Z1 = +2.13 + 2*0.02

im = plt.imshow(tomo[0], cmap='plasma',
                vmin=0., vmax=vmax,
                extent=[R0, R1, Z0, Z1],
                interpolation='bilinear',
                animated=True)

(x, y) = zip(*get_geom())
plt.plot(x, y, 'w', linewidth=0.5)

ticks = np.linspace(0., vmax, num=5)
labels = [str(t) for t in ticks]
labels[-1] = r'$\geq$' + labels[-1]
cb = plt.colorbar(im, fraction=0.26, ticks=ticks)
cb.ax.set_yticklabels(labels, fontsize=fontsize)
cb.ax.set_ylabel(r'MW m$^{-3}$', fontsize=fontsize)

fig = plt.gcf()
ax = plt.gca()

title = 'Pulse %s t=%.2fs' % (pulse, tomo_t[0])
ax.set_title(title, fontsize=fontsize)
ax.tick_params(labelsize=fontsize)
ax.set_xlabel('R (m)', fontsize=fontsize)
ax.set_ylabel('Z (m)', fontsize=fontsize)
ax.set_xlim([R0, R1])
ax.set_ylim([Z0, Z1])

plt.setp(ax.spines.values(), linewidth=0.1)
plt.tight_layout()

def animate(k):
    title = 'JET pulse %s t=%.2fs' % (pulse, tomo_t[k])
    ax.set_title(title, fontsize=fontsize)
    im.set_data(tomo[k])

animation = ani.FuncAnimation(fig, animate, frames=range(tomo.shape[0]))

fname = '%s_%.2f_%.2f.mp4' % (pulse, tomo_t[0], tomo_t[-1])
print 'Writing:', fname
animation.save(fname, fps=15, extra_args=['-vcodec', 'libx264'])
