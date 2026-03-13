"""
Multi-wavelength observation.
"""
import astropy.units as u
import math
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import glob, os
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.io import fits
from skimage import measure
from sunpy.coordinates import frames

file1 = 'data/manual_manual_align_Ha_r020/manual_manual_align_Ha_r020_20241003_062952.fits'
file2 = 'data/align_Ha_r040/align_Ha_r040_20241003_063003.fits'
file3 = 'data/align_manual_align_Ha_r060/align_manual_align_Ha_r060_20241003_063015.fits'
output_path = "output/overview.png"

# Generate simplest sunpy map
def create_map(data, crpix1, crpix2, crval1, crval2, scale, crota2, date_obs):
    coord_HIS = SkyCoord(crval1 * u.arcsec, crval2 * u.arcsec, obstime = date_obs, 
                         observer = 'earth', frame = frames.Helioprojective)
    header = sunpy.map.make_fitswcs_header(data, coord_HIS, 
                reference_pixel=[crpix1, crpix2] * u.pixel,
                scale = [scale, scale] * u.arcsec / u.pixel,
                rotation_angle = crota2 * u.degree)
    return sunpy.map.Map(data, header)

def read_fits(file):
    with fits.open(file) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    return data, header

# create axes positions
def create_axes(rows, cols, hmargin=0.02, vmargin=0.02, hspace=0.02, vspace=0.02):
    left_margin, right_margin = hmargin, hmargin
    bottom_margin, top_margin = vmargin, vmargin
    width = (1 - left_margin - right_margin - (cols-1)*hspace) / cols
    height = (1 - bottom_margin - top_margin - (rows-1)*vspace) / rows
    positions = []
    for row in range(rows):
        for col in range(cols):
            left = left_margin + col * (width + hspace)
            bottom = 1 - top_margin - (row + 1) * height - row * vspace
            positions.append([left, bottom, width, height])
    return positions

def main():
    data1, header1 = read_fits(file1)
    data2, header2 = read_fits(file2)
    data3, header3 = read_fits(file3)   
    map1 = create_map(data1, crpix1=(data1.shape[1]-1.0)/2.0, crpix2=(data1.shape[0]-1.0)/2.0, crval1=0, crval2=0, scale=0.165, crota2=0.0, date_obs=header1['DATE_OBS'])
    map2 = create_map(data2, crpix1=(data2.shape[1]-1.0)/2.0, crpix2=(data2.shape[0]-1.0)/2.0, crval1=0, crval2=0, scale=0.165, crota2=0.0, date_obs=header2['DATE_OBS'])
    map3 = create_map(data3, crpix1=(data3.shape[1]-1.0)/2.0, crpix2=(data3.shape[0]-1.0)/2.0, crval1=0, crval2=0, scale=0.165, crota2=0.0, date_obs=header3['DATE_OBS'])

    map_sequence = [map1, map2, map3]
    fig = plt.figure(figsize=(15, 10))
    positions = create_axes(rows=2, cols=3, hmargin=0.05, vmargin=0.05, hspace=0.05, vspace=0.05)
    axesA = [fig.add_axes(pos, projection=proj) for pos, proj in zip(positions[0:3], map_sequence)]
    map_sequence[0].plot(axes=axesA[0], norm=colors.Normalize(9000, 13000), cmap='afmhot')
    map_sequence[1].plot(axes=axesA[1], norm=colors.Normalize(7000, 10000), cmap='afmhot')
    map_sequence[2].plot(axes=axesA[2], norm=colors.Normalize(10000,13000), cmap='afmhot')
    axesB = [fig.add_axes(pos) for pos in positions[3:6]]
    axesB[0].hist(map_sequence[0].data.flatten(), bins=100, color='blue', alpha=0.7)
    axesB[1].hist(map_sequence[1].data.flatten(), bins=100, color='red', alpha=0.7)
    axesB[2].hist(map_sequence[2].data.flatten(), bins=100, color='green', alpha=0.7)

    plt.savefig(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()