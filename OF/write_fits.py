import astropy.units as u

import gc
import os
import sunpy.map
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from astropy.io import fits

class Config:
    def __init__(self):
        self.settings = {
            'har020': {
                'input': 'data/manual_manual_align_Ha_r020',
                'output': 'output/nvst/har020',
                'range': (5, 23), # 06:30 - 06:40，我们只处理这段时间的数据
                'coords': (-1.5, -4.5)  # 这里填入对应的 x1, y1
            },
            'har040': {
                'input': 'data/align_Ha_r040', 
                'output': 'output/nvst/har040',
                'range': (5, 23),
                'coords': (-6.8, 0.3) 
            },
            'har060': {
                'input': 'data/align_manual_align_Ha_r060',
                'output': 'output/nvst/har060',
                'range': (5, 23),
                'coords': (-10.7, -0.5)
            }
        }
config = Config()

def create_map(data, crpix1, crpix2, crval1, crval2, scale, crota2, date_obs):
    coord_HIS = SkyCoord(crval1 * u.arcsec, crval2 * u.arcsec, obstime = date_obs, 
                         observer = 'earth', frame = frames.Helioprojective)
    header = sunpy.map.make_fitswcs_header(data, coord_HIS, 
                reference_pixel=[crpix1, crpix2] * u.pixel,
                scale = [scale, scale] * u.arcsec / u.pixel,
                rotation_angle = crota2 * u.degree)
    return sunpy.map.Map(data, header)

def write_fits(file, output_path, x1, y1):
    with fits.open(file) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        map = create_map(data, crpix1=(data.shape[1]-1.0)/2.0, crpix2=(data.shape[0]-1.0)/2.0, crval1=x1, crval2=y1, scale=0.165, crota2=0.0, date_obs=header['DATE_OBS'])
        map.save(output_path, overwrite=True)
    
def main():
    for key, info in config.settings.items():
        input_dir = info['input']
        output_dir = info['output']
        start_idx, end_idx = info['range']
        x_val, y_val = info['coords']
        os.makedirs(output_dir, exist_ok=True)
        all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.fits')])
        files = all_files[start_idx:end_idx]
        print(f"Processing {key}: {len(files)} files, Coords: ({x_val}, {y_val})")
        for file in files:
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)
            write_fits(input_path, output_path, x1=x_val, y1=y_val)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
 