import os
from astropy.io import fits
from _generate_util import *

def get_crtsimg(fname):
    
    imgpath = os.path.join(get_data_path(), "CRTS_IMAGES", "CSS_trans_ims", fname)

    hdu_list = fits.open(imgpath)
    image_data = hdu_list[0].data
    hdu_list.close()
    return image_data

