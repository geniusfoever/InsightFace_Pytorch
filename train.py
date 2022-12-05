import os

from config import get_config
from Learner import face_learner
import argparse

# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=1, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument("-did", "--dataset_id", help="which img folder should be used", default=0, type=int)
    parser.add_argument('-lr','--lr',help='learning rate',default=1e-2, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=1024, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=6, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat]",default='ms1m', type=str)
    args = parser.parse_args()

    conf = get_config()
    
    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode
    conf.dataset_id = args.dataset_id%10 if args.dataset_id>-1 else 0
    learner = face_learner(conf)
    learner.load_state(conf,"ir_se50.pth" if args.dataset_id>-1 else ".pth",from_save_folder=True,model_only=(args.dataset_id<=-1),from_multiple_GPU=True)
    # learner.load_state(conf,"2022-11-27-18-42_accuracy-0.9362857142857143_step-1722_None.pth",False,False,True)
    print(conf.batch_size,"*"*10)
    learner.train(conf, args.epochs)


