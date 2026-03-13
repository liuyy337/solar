import sunpy.map
import os, glob
from astropy.io import fits

input_dir1 = 'data/align_Ha_r040'
input_dir2 = 'data/align_manual_align_Ha_r060'
input_dir3 = 'data/align_manual_Ha_r060'
input_dir4 = 'data/manual_manual_align_Ha_r020'

files1 = sorted(glob.glob(os.path.join(input_dir1, '*.fits')))
files2 = sorted(glob.glob(os.path.join(input_dir2, '*.fits')))
files3 = sorted(glob.glob(os.path.join(input_dir3, '*.fits')))
files4 = sorted(glob.glob(os.path.join(input_dir4, '*.fits')))

# map = sunpy.map.Map(files[0])
# print(map.meta)

with fits.open(files1[0]) as hdul:
    print(hdul.info())
    print(hdul[0].header.cards)
    # print('data shape = ', hdul[0].data.shape)

# data1.shape = (1066, 1160)
# data2.shape = (1278, 1242)
# data3.shape = (1278, 1242)
# data4.shape = (1356, 1322)