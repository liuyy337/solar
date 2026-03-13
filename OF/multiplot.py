import astropy.units as u
import cv2
import gc
import glob, os
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import sunpy.map
from astropy.coordinates import SkyCoord
from multiprocessing import Pool, cpu_count
from sunpy.coordinates import frames

class Config():
    def __init__(self):
        self.image_folders = [
            'output/nvst/har020',
            'output/nvst/har040',
            'output/nvst/har060',
        ]
        self.image_configs = [
            (colors.Normalize(9000, 13000),  'afmhot', 10, 5, True,  True,   "(a) Ha + 0.2 "), 
            (colors.Normalize(7000, 10000),  'afmhot', 10, 5, True,  False,  "(b) Ha + 0.4 "), 
            (colors.Normalize(10000,13000),  'afmhot', 10, 5, True,  False,  "(c) Ha + 0.6 "), 
        ]
        self.output_dir = "output/aligned_nvst"
        self.xc, self.yc = -25, -50
        self.dx, self.dy = 50, 50
config = Config()

def create_map(data, crpix1, crpix2, crval1, crval2, scale, crota2, date_obs):
    coord_HIS = SkyCoord(crval1 * u.arcsec, crval2 * u.arcsec, obstime = date_obs, 
                         observer = 'earth', frame = frames.Helioprojective)
    header = sunpy.map.make_fitswcs_header(data, coord_HIS, 
                reference_pixel=[crpix1, crpix2] * u.pixel,
                scale = [scale, scale] * u.arcsec / u.pixel,
                rotation_angle = crota2 * u.degree)
    return sunpy.map.Map(data, header)

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

def _setup_coord(coord, axis, spacing, minors, visible, labelsize):
    coord.set_ticks_position('bltr')
    coord.set_axislabel(f'{axis} (arcsec)', fontsize=labelsize)
    coord.set_ticks(spacing=spacing * u.arcsec)
    coord.set_format_unit(u.arcsec, show_decimal_unit=False)
    coord.display_minor_ticks(True)
    coord.set_minor_frequency(minors)
    coord.set_ticklabel_visible(visible)

# Configure sunpy map
def setup_map(ax, spacing, minors, grid=True, xaxis=True, yaxis=True, 
              string1='', string2='', textcolor='white', fontsize=16, labelsize=12):
    ax.set_title('')
    ax.text(0.04, 0.96, string1, transform=ax.transAxes, color=textcolor, fontsize=fontsize, verticalalignment='top')
    ax.text(0.04, 0.10, string2, transform=ax.transAxes, color=textcolor, fontsize=fontsize, verticalalignment='top')
    lon, lat = ax.coords
    _setup_coord(lon, 'X', spacing, minors, xaxis, labelsize=labelsize)
    _setup_coord(lat, 'Y', spacing, minors, yaxis, labelsize=labelsize)
    ax.tick_params(axis='both', which='major', length=8, direction='in', color='black', labelsize=labelsize)
    ax.tick_params(axis='both', which='minor', length=4)
    ax.coords.grid(linestyle='--', linewidth=1.0) if grid else ax.coords.grid(False)

def plot_images(arg):
    files, output_path = arg
    map_sequence = []
    for f in files:
        m = sunpy.map.Map(f)
        bottom_left = SkyCoord((config.xc - config.dx/2)*u.arcsec, (config.yc - config.dy/2)*u.arcsec, frame=m.coordinate_frame)
        top_right = SkyCoord((config.xc + config.dx/2)*u.arcsec, (config.yc + config.dy/2)*u.arcsec, frame=m.coordinate_frame)
        m = m.submap(bottom_left, top_right=top_right)
        map_sequence.append(m)

    # map_sequence = sunpy.map.Map(files)
    # data1 = map_sequence[0].data[250:550, 400:700]
    # data2 = map_sequence[1].data[77:377, 350:650]
    # data3 = map_sequence[2].data[187:487, 420:720]
    # map1 = create_map(data1, crpix1=(data1.shape[1]-1.0)/2.0, crpix2=(data1.shape[0]-1.0)/2.0, crval1=0, crval2=0, scale=0.165, crota2=0.0, date_obs=map_sequence[0].date)
    # map2 = create_map(data2, crpix1=(data2.shape[1]-1.0)/2.0, crpix2=(data2.shape[0]-1.0)/2.0, crval1=0, crval2=0, scale=0.165, crota2=0.0, date_obs=map_sequence[1].date)
    # map3 = create_map(data3, crpix1=(data3.shape[1]-1.0)/2.0, crpix2=(data3.shape[0]-1.0)/2.0, crval1=0, crval2=0, scale=0.165, crota2=0.0, date_obs=map_sequence[2].date)
    # map_sequence = [map1, map2, map3]

    fig = plt.figure(figsize=(15, 5), dpi=100)
    positions = create_axes(rows=1, cols=3, hmargin=0.06, vmargin=0.06, hspace=0.002, vspace=0.002)
    axes = [fig.add_axes(pos, projection=proj) for pos, proj in zip(positions, map_sequence)]
    for i, conf in enumerate(config.image_configs):
        map_sequence[i].plot(axes=axes[i], norm=conf[0], cmap=conf[1])
        setup_map(axes[i], spacing=conf[2], minors=conf[3], grid=False, xaxis=conf[4], yaxis=conf[5], string1=conf[6]+map_sequence[i].date.strftime('%H:%M:%S'), fontsize=16, labelsize=14)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)
    del map_sequence, fig, axes

def image2movie(input_dir, fps):
    files = sorted(glob.glob(input_dir + '/*.png'))
    output_path = input_dir + ".mp4"
    cv2_fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame = cv2.imread(files[0])
    height, width, layers = frame.shape
    video  = cv2.VideoWriter(output_path, cv2_fourcc, fps, (width, height))
    for file in files:
        video.write(cv2.imread(file))
    cv2.destroyAllWindows()
    video.release()
    print(f"Saved: {output_path}")

def multiplot():
    os.makedirs(config.output_dir, exist_ok=True)
    lists = [sorted(glob.glob(os.path.join(d, '*.fits'))) for d in config.image_folders]
    task_args = []
    for i in range(18):
        files = (lists[0][i], lists[1][i], lists[2][i])
        task_args.append((files, os.path.join(config.output_dir, f"frame_{i:02d}.png")))
    with Pool(processes=cpu_count() - 1) as pool:
        pool.map(plot_images, task_args)

if __name__ == "__main__":
    multiplot()
    image2movie(input_dir=config.output_dir, fps=2)