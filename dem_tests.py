import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import elevation
import richdem as rd

matplotlib.rcParams['figure.figsize'] = (8, 5.5)


if __name__ == '__main__':
    elevation.clip(bounds=(12.35, 41.8, 12.65, 42), output='Rome-DEM.tif')
    # dem_path = os.path.join(os.getcwd(), 'Shasta-30m-DEM.tif')
    # elevation.clip(bounds=(-122.4, 41.2, -122.1, 41.5), output=dem_path)

    # shasta_dem = rd.LoadGDAL(dem_path)
    # plt.imshow(shasta_dem, interpolation='none')
    # plt.colorbar()
    # plt.show()
