import os
import tarfile
from astropy.io import fits
from _generate_util import *

def get_crtsimg(fname):

    p = os.path.join(get_data_path(), "CRTS_IMAGES")
    imgpath = os.path.join(p, "CSS_trans_ims")

    if not os.path.isdir(imgpath):

        fnamezip = os.path.join(p, "CSS_trans_ims.tar.gz")
        check_and_download(fnamezip)

        print("Extracting content ...")
        tar = tarfile.open(fnamezip, "r:gz")
        tar.extractall(path=imgpath)
        tar.close()

    hdu_list = fits.open(os.path.join(imgpath, fname))
    image_data = hdu_list[0].data
    hdu_list.close()
    return image_data

