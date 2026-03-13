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
output_path = "output/alignment.png"

# x1, y1 = -1.5, -4.5
# x2, y2 = -6.8, 0.3
# x3, y3 = -10.7, -0.5

x1, y1 = 0, 0
x2, y2 = 0, 0
x3, y3 = 0, 0

xc, yc = -25, -50
dx, dy = 50, 50

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
    # data1 = data1[250:550, 400:700]
    # data2 = data2[77:377, 350:650]
    # data3 = data3[187:487, 420:720]

    map1 = create_map(data1, crpix1=(data1.shape[1]-1.0)/2.0, crpix2=(data1.shape[0]-1.0)/2.0, crval1=x1, crval2=y1, scale=0.165, crota2=0.0, date_obs=header1['DATE_OBS'])
    map2 = create_map(data2, crpix1=(data2.shape[1]-1.0)/2.0, crpix2=(data2.shape[0]-1.0)/2.0, crval1=x2, crval2=y2, scale=0.165, crota2=0.0, date_obs=header2['DATE_OBS'])
    map3 = create_map(data3, crpix1=(data3.shape[1]-1.0)/2.0, crpix2=(data3.shape[0]-1.0)/2.0, crval1=x3, crval2=y3, scale=0.165, crota2=0.0, date_obs=header3['DATE_OBS'])
    bottom_left = SkyCoord((xc - dx/2) * u.arcsec, (yc - dy/2) * u.arcsec, frame=map1.coordinate_frame)
    top_right = SkyCoord((xc + dx/2) * u.arcsec, (yc + dy/2) * u.arcsec, frame=map1.coordinate_frame)
    map1 = map1.submap(bottom_left, top_right=top_right)
    map2 = map2.submap(bottom_left, top_right=top_right)
    map3 = map3.submap(bottom_left, top_right=top_right)

    map_sequence = [map1, map2, map3]
    fig = plt.figure(figsize=(15, 5))
    positions = create_axes(rows=1, cols=3, hmargin=0.05, vmargin=0.05, hspace=0.07, vspace=0.05)
    axes = [fig.add_axes(pos, projection=proj) for pos, proj in zip(positions, map_sequence)]
    map_sequence[0].plot(axes=axes[0], norm=colors.Normalize(9000, 13000), cmap='afmhot')
    map_sequence[1].plot(axes=axes[1], norm=colors.Normalize(7000, 10000), cmap='afmhot')
    map_sequence[2].plot(axes=axes[2], norm=colors.Normalize(10000,13000), cmap='afmhot')

    plt.savefig(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()