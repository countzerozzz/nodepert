import csv
import numpy as np
import pandas as pd
import pickle
import argparse
from datetime import datetime, timedelta

def file_writer(dir_path, expdata, meta_data):
    update_rule, n_hl, lr, batchsize, hl_size, num_epochs, elapsed_time = meta_data
    params = ["update_rule",update_rule.upper(),"depth", str(n_hl), "width", str(hl_size), "lr", str(lr),
              "num_epochs", str(num_epochs), "batchsize",str(batchsize)]

    output = open(dir_path, 'wb')
    pickle.dump(expdata, output)
    output.close()
    return

def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def sma_accuracy(arr, period):
    return np.ma.average(arr[-int(period):])

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-network", type=str, default='fc')
    ap.add_argument("-update_rule", type=str, default='sgd')
    ap.add_argument("-n_hl", type=int, default=2)
    ap.add_argument("-lr", type=float, default=1e-2)
    ap.add_argument("-batchsize", type=int, default=100)
    ap.add_argument("-hl_size", type=int, default=500)
    ap.add_argument("-num_epochs", type=int, default=10)
    ap.add_argument('-log_expdata', type=str_to_bool, nargs='?', const=True, default=False)
    args= ap.parse_args()

    return args.network, args.update_rule, args.n_hl, args.lr, args.batchsize, args.hl_size, args.num_epochs, args.log_expdata

def get_elapsed_time(sec):
    sec=timedelta(seconds=int(sec))
    d = datetime(1,1,1) + sec
    if(d.hour>0):
        return str(d.hour)+"hours"+ str(d.minute)+"min "+ str(d.second)+"sec"
    else:
        return str(d.minute)+"min "+ str(d.second)+"sec"
    