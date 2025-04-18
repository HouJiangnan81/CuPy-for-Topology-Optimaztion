import numpy as np
def conv(radius):
    #center = (radius ,radius)
    array = np.zeros((2*radius+1, 2*radius+1, 2*radius+1))
    x, y, z = np.indices(array.shape)
    distances_squared = (x - radius)**2 + (y - radius)**2 + (z - radius)**2
    array[distances_squared <= radius**2] = 1
    #array[distances_squared <= 0.25*(radius ** 2)] = 2
    return array