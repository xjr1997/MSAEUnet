# main program for dispersion picker
import os
import yaml
import argparse

from SEA.train import train
from SEA.test import test
from SEA.eval import eval

def SEA_main(args):
    """
    Multi-Scale Attention-Enhanced Deep Learning Model for Ionogram Automatic Scaling
    """
    # load configuration file
    cfgs = yaml.load(open(args.config_file), Loader=yaml.BaseLoader)

    # set gpu id
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu_id)

    # check if train or test
    if not (args.run_train or args.run_test or args.run_eval):
        print ('Please set one of the options --train | --test')
        parser.print_help()
        return
    
    # train
    if args.run_train:
        print('Setting Trainer')
        train(cfgs)
        print('Done Training')
        return
    
    # test
    if args.run_test:
        print('Setting Tester')
        test(cfgs)
        print('Done Testing')
        return
    
    # evaluate
    if args.run_eval:
        print('Start Evaluation')
        eval(cfgs)
        print('Done Evaluation')
        return
    
    return

if __name__ == '__main__':
    # prase config data
    parser = argparse.ArgumentParser(description='Multi-Scale Attention-Enhanced Deep Learning Model for Ionogram Automatic Scaling')
    parser.add_argument('--config-file', dest='config_file', type=str, help='Experiment configuration file')
    parser.add_argument('--train', dest='run_train', action='store_true', default=False, help='Launch training')
    parser.add_argument('--test', dest='run_test', action='store_true', default=False, help='Launch testing on a list of Ionograms')
    parser.add_argument('--eval', dest='run_eval', action='store_true', default=False, help='Launch evaluation on a list of Ionograms')
    parser.add_argument('--gpuid',dest='gpu_id',type=int, default=0, help='Run On a certain GPU')

    args = parser.parse_args()
    
    dias_main(args)        
