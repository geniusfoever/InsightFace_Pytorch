from pathlib import Path
from config import get_config
from data.data_mxnet import load_bin, load_mx_rec
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-r", "--rec_path", help="mxnet record file path",default=r"E:\dataset\celeba\celeba", type=str)
    args = parser.parse_args()
    conf = get_config()
    rec_path = args.rec_path
    # load_mx_rec(rec_path)

    # bin_files = ['lfw']#'agedb_30', 'cfp_fp', ] #, 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
    #
    # for i in range(len(bin_files)):
    #     load_bin(rec_path/(bin_files[i]+'.bin'), rec_path/bin_files[i], conf.test_transform)

    load_bin(Path(rec_path+'.bin'), Path(rec_path), conf.test_transform)