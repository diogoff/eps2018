# Source code for the EPS 2018 paper

[Regularization extraction for real-time plasma tomography at JET](http://web.tecnico.ulisboa.pt/diogo.ferreira/papers/ferreira18regularization.pdf)  
D. R. Ferreira, D. D. Carvalho, P. J. Carvalho, H. Fernandes, and JET Contributors  
[45th EPS Conference on Plasma Physics](https://eps2018.eli-beams.eu/en/), Prague, Czech Republic, July 2-6, 2018

## Files

* `geam.txt` and `kb5_los.txt` contain the geometry for the vessel and the KB5 lines of sight, respectively.

* `geom.py` contains some simple routines to read those geometry files.

* `tomo_kb5_reliable.hdf` contais the reconstructions that were used for data fiting; `bolo_kb5_reliable.hdf` contains the full KB5 signals for the same pulses.

* `fit_M.py` fits the *M* matrix using the full dataset; `fit_validate_M.py` fits *M* using 90% for training and 10% for validation.

* `plot_train.py` plots the loss and validation loss during (or after) training.

* When training finishes, *M* will be saved to `M.npy`.


