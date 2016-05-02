import os
import tarfile
from astropy.io import fits
from _generate_util import *

def get_crtsimg(fname, subimage="master"):

    p = os.path.join(get_data_path(), "CRTS_IMAGES")
    imgpath = os.path.join(p, "CSS_trans_ims")

    if not os.path.isdir(imgpath):

        fnamezip = os.path.join(p, "CSS_trans_ims.tar.gz")
        check_and_download(fnamezip)

        print("Extracting content ...")
        tar = tarfile.open(fnamezip, "r:gz")
        tar.extractall(path=imgpath)
        tar.close()

    if subimage == "master":
        fname = fname + ".master.fits"
    else:
        fname = fname + "-000%i.arch.fits" % int(subimage)

    hdu_list = fits.open(os.path.join(imgpath, fname))
    image_data = hdu_list[0].data
    hdu_list.close()

    return image_data

def get_crtsimg_allnames():

    p = os.path.join(get_data_path(), "CRTS_IMAGES")
    imgpath = os.path.join(p, "CSS_trans_ims")

    allfiles = os.listdir(imgpath)

    filenames = []
    for i in xrange(len(allfiles)):
        fname = allfiles[i].split(".")[0].split("-")[0]
        filenames.append(fname)

    filenames = list(set(filenames))

    return filenames
        
