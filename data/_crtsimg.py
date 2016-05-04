import os
import tarfile
from astropy.io import fits
from _generate_util import *
from pandas import read_html, read_csv

def _split_label(lab):

    try:
        res = lab.split(" ")[0]
    except:
        res = "NaN"
    
    return res

def _get_labels():

    p = os.path.join(get_data_path(), "CRTS_IMAGES")
    labelfile = os.path.join(p, "labels.csv")
    if not os.path.isfile(labelfile):
        allns_html = os.path.join(p, "Allns.html")
        check_and_download(allns_html)

        print("Extracting labels ...")
        labels = read_html(allns_html, index_col=5, header=0)[0]
        labels = labels[['Classification']]

        labels = labels.applymap(_split_label)
        labels.to_csv(labelfile)
    
    labels = read_csv(labelfile)

    # two times the entry for '1310120180014101233'
    labels = labels.drop_duplicates()
    labels = labels.set_index("CSS images")

    labeldict = {}
    for index, row in labels.iterrows():
        labeldict[str(index)] =  row.values[0]

    return labeldict

def _get_crtsimg(fname, subimage="master"):

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

    try:
        label = labels[fname.split(".")[0]]
    except:
        label = "UNKNOWN"

    return image_data

def _get_crtsimg_allnames():

    p = os.path.join(get_data_path(), "CRTS_IMAGES")
    imgpath = os.path.join(p, "CSS_trans_ims")

    allfiles = os.listdir(imgpath)

    filenames = []
    for i in xrange(len(allfiles)):
        fname = allfiles[i].split(".")[0].split("-")[0]
        filenames.append(fname)

    filenames = list(set(filenames))

    return filenames

def get_crtsimages(fnames=None):

    if fnames is None:
        fnames = _get_crtsimg_allnames()
    labels = _get_labels()

    X = []
    y = []
    I = []

    for i in xrange(len(fnames)):

        fname = fnames[i]

        try:

            label = labels[fname]

            Ximg = _get_crtsimg(fname, subimage="master")
            for i in xrange(1,5):
                img = _get_crtsimg(fname, subimage=i)
                Ximg = numpy.dstack((Ximg, img))

            X.append(Ximg)
            y.append(label)
            I.append(fname)

        except Exception as e:
            print "Problem parsing %s: %s" % (str(fname), str(e))

    return X, y, I
        
