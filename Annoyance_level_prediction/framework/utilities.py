import os
import numpy as np


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


import h5py
def calculate_scalar(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    ################### save #######################################################################
    # Create hdf5 file
    hdf5_path = os.path.join(os.getcwd(), 'scalar.h5')
    hf = h5py.File(hdf5_path, 'w')

    hf.create_dataset(name='mean', data=mean,
                      dtype=np.float32)

    hf.create_dataset(name='std',
                      data=std,
                      dtype=np.float32)

    hf.close()

    with h5py.File(hdf5_path, 'r') as hf:
        mean = hf['mean'][:]
        std = hf['std'][:]
    print('mean shape: ', mean.shape)
    print('std shape: ', std.shape)


    return mean, std


def scale(x, mean, std):
    return (x - mean) / std








