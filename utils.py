import csv
import numpy as np
import argparse
from datetime import datetime, timedelta

def file_writer(dir_path, data, meta_data, norms=None, write_norms=False):
    update_rule, n_hl, lr, batchsize, hl_size, n_epochs, elapsed_time = meta_data
    params = ["update_rule",update_rule.upper(),"depth", str(n_hl), "width", str(hl_size), "lr", str(lr),
              "n_epochs", str(n_epochs), "batchsize",str(batchsize)]

    with open(dir_path, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(params)
        writer.writerow("")

        if(write_norms):
            for hl in range(n_hl+1):
                writer.writerow(norms[str(hl)])
        
        writer.writerow(data)
        writer.writerow(str(elapsed_time))
        writer.writerow("")
        csvFile.flush()

    csvFile.close()
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
    ap.add_argument("-update_rule", type=str, default='np')
    ap.add_argument("-n_hl", type=int, default=2)
    ap.add_argument("-lr", type=float, default=1e-3)
    ap.add_argument("-batchsize", type=int, default=100)
    ap.add_argument("-hl_size", type=int, default=500)
    ap.add_argument("-n_epochs", type=int, default=10)
    ap.add_argument('-write_file', type=str_to_bool, nargs='?', const=True, default=False)
    args= ap.parse_args()

    return args.update_rule, args.n_hl, args.lr, args.batchsize, args.hl_size, args.n_epochs

def get_elapsed_time(sec):
    sec=timedelta(seconds=int(sec))
    d = datetime(1,1,1) + sec
    if(d.hour>0):
        return str(d.hour)+"hours"+ str(d.minute)+"min "+ str(d.second)+"sec"
    else:
        return str(d.minute)+"min "+ str(d.second)+"sec"
    