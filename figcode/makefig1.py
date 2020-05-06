import npimports
from npimports import *

import matplotlib

path = "explogs/fig1exp/"

npexpdata = pickle.load(open(path + "npexpdata.pickle", "rb"))
npparams = pickle.load(open(path + "npparams.pickle", "rb"))

sgdexpdata = pickle.load(open(path + "sgdexpdata.pickle", "rb"))
sgdparams = pickle.load(open(path + "sgdparams.pickle", "rb"))
