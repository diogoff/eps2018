
import numpy as np

# -------------------------------------------------------------------------

def get_geom():
    fname = 'geom.txt'
    print 'Reading:', fname
    f = open(fname, 'r')
    geom = []
    for line in f:
        parts = line.split()
        if len(parts) == 2:
            r = float(parts[0])
            z = float(parts[1])
            geom.append((r, z))
    f.close()
    return geom

# -------------------------------------------------------------------------

n_rows = 196
n_cols = 115

r_min = +1.71
r_max = +3.99

z_min = -1.77
z_max = +2.13

def transform(r, z):
    i = int(round((z_max-z)/(z_max-z_min)*(n_rows-1)))
    j = int(round((r-r_min)/(r_max-r_min)*(n_cols-1)))
    return (i, j)

# -------------------------------------------------------------------------

def get_kb5_los():
    fname = 'kb5_los.txt'
    print 'Reading:', fname
    f = open(fname, 'r')
    kb5_los = []
    for line in f:
        parts = line.split()
        if len(parts) == 22:
            Rst = float(parts[10])
            Zst = float(parts[11])
            Rend = float(parts[13])
            Zend = float(parts[14])
            kb5_los.append((Rst, Zst, Rend, Zend))
    f.close()
    return kb5_los
