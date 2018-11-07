import numpy as np
import healpy as hp
import pandas
import camb
import ispice
from astropy.table import Table
from astropy.io import fits
from scipy import integrate
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.special import spherical_jn
from scipy.stats import norm
from scipy.optimize import curve_fit
from functools import partial
import inspect
import copy


speedOfLightSI = 299792458.0
MpcInMeters = 3.0856775814671917e22
GyrInSeconds = 3.15576e16

speedOfLightMpcGyr = speedOfLightSI/MpcInMeters*GyrInSeconds

CMBTemp=2.726

from sphericosmo.cosmocontainer import *
from sphericosmo.sphericalpower import *
from sphericosmo.pitau import *
from sphericosmo.clcontainer import *
from sphericosmo.mapcontainer import *
from sphericosmo.redshiftcounthist import *
from sphericosmo.theoryclcomputer import *
from sphericosmo.spicecorrelator import *
from sphericosmo.clfitter import *
from sphericosmo.fisherfitter import *

