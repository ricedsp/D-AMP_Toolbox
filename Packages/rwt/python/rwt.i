%define MODDOCSTRING
"The Rice Wavelet Toolbox (RWT) is a collection of functions for 1D and 2D 
wavelet and filter bank design, analysis, and processing."
%enddef

%module(docstring=MODDOCSTRING) rwt

/* The C functions for the transforms are not suitable for direct use from python so let's rename them. */

%rename(_c_dwt)     dwt;
%rename(_c_idwt)   idwt;
%rename(_c_rdwt)   rdwt;
%rename(_c_irdwt) irdwt;

%rename(_find_levels) rwt_find_levels;
%rename(_check_levels) rwt_check_levels;

%{
  #define SWIG_FILE_WITH_INIT
  #include "../lib/inc/rwt_transforms.h"
  #include "../lib/inc/rwt_init.h"
%}

%include "../lib/inc/rwt_init.h"
%include "numpy.i"

%init %{
  import_array();
%}

/* Building on the numpy SWIG macros we make wrapper functions for 1D and 2D for each transform */

void _c_dwt_1(  double* INPLACE_ARRAY1, int DIM1,           double* INPLACE_ARRAY1, int DIM1, int levels, double* INPLACE_ARRAY1, int DIM1);
void _c_dwt_2(  double* INPLACE_ARRAY2, int DIM1, int DIM2, double* INPLACE_ARRAY1, int DIM1, int levels, double* INPLACE_ARRAY2, int DIM1, int DIM2);
void _c_idwt_1( double* INPLACE_ARRAY1, int DIM1,           double* INPLACE_ARRAY1, int DIM1, int levels, double* INPLACE_ARRAY1, int DIM1);
void _c_idwt_2( double* INPLACE_ARRAY2, int DIM1, int DIM2, double* INPLACE_ARRAY1, int DIM1, int levels, double* INPLACE_ARRAY2, int DIM1, int DIM2);
void _c_rdwt_1( double* INPLACE_ARRAY1, int DIM1,           double* INPLACE_ARRAY1, int DIM1, int levels, double* INPLACE_ARRAY1, int DIM1,           double* INPLACE_ARRAY1, int DIM1);
void _c_rdwt_2( double* INPLACE_ARRAY2, int DIM1, int DIM2, double* INPLACE_ARRAY1, int DIM1, int levels, double* INPLACE_ARRAY2, int DIM1, int DIM2, double* INPLACE_ARRAY2, int DIM1, int DIM2);
void _c_irdwt_1(double* INPLACE_ARRAY1, int DIM1,           double* INPLACE_ARRAY1, int DIM1, int levels, double* INPLACE_ARRAY1, int DIM1,           double* INPLACE_ARRAY1, int DIM1);
void _c_irdwt_2(double* INPLACE_ARRAY2, int DIM1, int DIM2, double* INPLACE_ARRAY1, int DIM1, int levels, double* INPLACE_ARRAY2, int DIM1, int DIM2, double* INPLACE_ARRAY2, int DIM1, int DIM2);

%inline %{

void _c_dwt_1(double *x, int nrows, double *h, int ncoeff, int levels, double *y, int toss1) {
  dwt(x, nrows, 1, h, ncoeff, levels, y);
}

void _c_idwt_1(double *x, int nrows, double *h, int ncoeff, int levels, double *y, int toss1) {
  idwt(x, nrows, 1, h, ncoeff, levels, y);
}

void _c_rdwt_1(double *x, int nrows, double *h, int ncoeff, int levels, double *yl, int toss1, double *yh, int toss2) {
  rdwt(x, nrows, 1, h, ncoeff, levels, yl, yh);
}

void _c_irdwt_1(double *x, int nrows, double *h, int ncoeff, int levels, double *yl, int toss1, double *yh, int toss2) {
  irdwt(x, nrows, 1, h, ncoeff, levels, yl, yh);
}

void _c_dwt_2(double *x, int nrows, int ncols, double *h, int ncoeff, int levels, double *y, int toss1, int toss2) {
  dwt(x, nrows, ncols, h, ncoeff, levels, y);
}

void _c_idwt_2(double *x, int nrows, int ncols, double *h, int ncoeff, int levels, double *y, int toss1, int toss2) {
  idwt(x, nrows, ncols, h, ncoeff, levels, y);
}

void _c_rdwt_2(double *x, int nrows, int ncols, double *h, int ncoeff, int levels, double *yl, int toss1, int toss2, double *yh, int toss3, int toss4) {
  rdwt(x, nrows, ncols, h, ncoeff, levels, yl, yh);
}

void _c_irdwt_2(double *x, int nrows, int ncols, double *h, int ncoeff, int levels, double *yl, int toss1, int toss2, double *yh, int toss3, int toss4) {
  irdwt(x, nrows, ncols, h, ncoeff, levels, yl, yh);
}

%}

%pythoncode %{

import numpy as np

def _levels(x, L):
  dim = len(x.shape) # Determine the dimensions of our input
  m = x.shape[0]
  if (dim == 2):
    n = x.shape[1]
  else:
    n = 1
  if (L == 0): # If the number of levels was not specified then use the max
    L = _find_levels(m, n)
  _check_levels(L, m, n) # Sanity check the number of levels
  return L

def dwt(x, h, L = 0):
  """
Function computes the discrete wavelet transform y for a 1D or 2D input
signal x using the scaling filter h.

Input:
   x : finite length 1D or 2D signal (implicitly periodized)
   h : scaling filter
   L : number of levels. In the case of a 1D signal, length(x) must be
       divisible by 2^L; in the case of a 2D signal, the row and the
       column dimension must be divisible by 2^L. If no argument is
       specified, a full DWT is returned for maximal possible L.

Output:
   y : the wavelet transform of the signal 
       (see example to understand the coefficients)
   L : number of decomposition levels

1D Example:
   x = makesig('LinChirp', 8)
   h = daubcqf(4, 'min')[0]
   L = 2
   y,L = dwt(x,h,L)

1D Example's output and explanation:

   y = array([1.1097,0.8767,0.8204,-0.5201,-0.0339,0.1001,0.2201,-0.1401])
   L = 2

The coefficients in output y are arranged as follows

   y(0) and y(1) : Scaling coefficients (lowest frequency)
   y(2) and y(3) : Band pass wavelet coefficients
   y(4) to y(7)  : Finest scale wavelet coefficients (highest frequency)

2D Example:

   load test_image        
   h = daubcqf(4,'min')[0]
   L = 1
   y,L = dwt(test_image,h,L)

2D Example's output and explanation:

   The coefficients in y are arranged as follows.

          .------------------.
          |         |        |
          |    4    |   2    |
          |         |        |
          |   L,L   |   H,L  |
          |         |        |
          --------------------
          |         |        |
          |    3    |   1    |
          |         |        |
          |   L,H   |  H,H   |
          |         |        |
          `------------------'
   
   where 
        1 : High pass vertically and high pass horizontally
        2 : Low pass vertically and high pass horizontally
        3 : High pass vertically and low  pass horizontally
        4 : Low pass vertically and Low pass horizontally 
            (scaling coefficients)
  """
  if (x.dtype != 'float'):
    x = x * 1.0
  L = _levels(x, L)
  y = np.zeros(x.shape)
  dim = len(x.shape)
  if (dim == 1):
    _rwt._c_dwt_1(x, h, L, y)
  if (dim == 2):
    _rwt._c_dwt_2(x, h, L, y)
  return y, L

def idwt(y, h, L = 0):
  """
Function computes the inverse discrete wavelet transform x for a 1D or
2D input signal y using the scaling filter h.

Input:
   y : finite length 1D or 2D input signal (implicitly periodized)
       (see function mdwt to find the structure of y)
   h : scaling filter
   L : number of levels. In the case of a 1D signal, length(x) must be
       divisible by 2^L; in the case of a 2D signal, the row and the
       column dimension must be divisible by 2^L.  If no argument is
       specified, a full inverse DWT is returned for maximal possible
       L.

Output:
   x : periodic reconstructed signal
   L : number of decomposition levels

1D Example:
   xin = makesig('LinChirp', 8)
   h = daubcqf(4, 'min')[0]
   L = 1
   y, L = mdwt(xin, h, L)
   x, L = midwt(y, h, L)

1D Example's output:

   x = array([0.0491,0.1951,0.4276,0.7071,0.9415,0.9808,0.6716,0.0000])
   L = 1
  """
  if (y.dtype != 'float'):
    y = y * 1.0
  L = _levels(y, L)
  x = np.zeros(y.shape)
  dim = len(x.shape)
  if (dim == 1):
    _rwt._c_idwt_1(x, h, L, y)
  if (dim == 2):
    _rwt._c_idwt_2(x, h, L, y)
  return x, L

def rdwt(x, h, L = 0):
  """
Function computes the redundant discrete wavelet transform y
for a 1D  or 2D input signal. (Redundant means here that the
sub-sampling after each stage is omitted.) yl contains the
lowpass and yh the highpass components. In the case of a 2D
signal, the ordering in yh is 
[lh hl hh lh hl ... ] (first letter refers to row, second to
column filtering). 

Input:
   x : finite length 1D or 2D signal (implicitly periodized)
   h : scaling filter
   L : number of levels. In the case of a 1D 
       length(x) must be  divisible by 2^L;
       in the case of a 2D signal, the row and the
       column dimension must be divisible by 2^L.
       If no argument is
       specified, a full DWT is returned for maximal possible L.

Output:
   yl : lowpass component
   yh : highpass components
   L  : number of levels

Example:
  x = makesig('Leopold', 8)
  h = daubcqf(4, 'min')[0]
  L = 1
  yl, yh, L = mrdwt(x,h,L)

Example's output:
  yl =  0.8365  0.4830 0 0 0 0 -0.1294 0.2241
  yh = -0.2241 -0.1294 0 0 0 0 -0.4830 0.8365
  L = 1
  """
  if (x.dtype != 'float'):
    x = x * 1.0
  L = _levels(x, L)
  yl = np.zeros(x.shape)
  dim = len(x.shape)
  if (dim == 1):
    yh = np.zeros(x.shape[0] * L)
    _rwt._c_rdwt_1(x, h, L, yl, yh)
  if (dim == 2):
    yh = np.zeros((x.shape[0], x.shape[1] * L * 3))
    _rwt._c_rdwt_2(x, h, L, yl, yh)
  return yl, yh, L

def irdwt(yl, yh, h, L = 0):
  """
Function computes the inverse redundant discrete wavelet
transform x  for a 1D or 2D input signal. (Redundant means here
that the sub-sampling after each stage of the forward transform
has been omitted.) yl contains the lowpass and yl the highpass
components as computed, e.g., by mrdwt. In the case of a 2D
signal, the ordering in
yh is [lh hl hh lh hl ... ] (first letter refers to row, second
to column filtering).  

Input:
   yl : lowpass component
   yh : highpass components
   h  : scaling filter
   L  : number of levels. In the case of a 1D signal, 
        length(yl) must  be divisible by 2^L;
        in the case of a 2D signal, the row and
        the column dimension must be divisible by 2^L.

Output:
        x : finite length 1D or 2D signal
        L : number of levels

Example:
  xin = makesig('Leopold', 8)
  h = daubcqf(4, 'min')[0]
  L = 1
  yl, yh, L = mrdwt(xin, h, L)
  x, L = mirdwt(yl, yh, h, L)

Example Output:
  x = array([0.0000,1.0000,0.0000,-0.0000,0,0,0,-0.0000])
  L = 1
  """
  if (yl.dtype != 'float'):
    yl = yl * 1.0
  if (yh.dtype != 'float'):
    yh = yh * 1.0
  L = _levels(yl, L)
  x = np.zeros(yl.shape)
  dim = len(x.shape)
  if (dim == 1):
    _rwt._c_irdwt_1(x, h, L, yl, yh)
  if (dim == 2):
    _rwt._c_irdwt_2(x, h, L, yl, yh)
  return x, L

def daubcqf(n, dtype = 'min'):
  """
Function computes the Daubechies' scaling and wavelet filters
(normalized to sqrt(2)).

Input: 
   n     : Length of filter (must be even)
   dtype : Optional parameter that distinguishes the minimum phase,
           maximum phase and mid-phase solutions ('min', 'max', or
           'mid'). If no argument is specified, the minimum phase
           solution is used.

Output: 
   h_0 : Minimal phase Daubechies' scaling filter 
   h_1 : Minimal phase Daubechies' wavelet filter 

Example:
   n = 4
   dtype = 'min'
   h_0, h_1 = daubcqf(n, dtype)

Example Result:
   h_0 = array([0.4830, 0.8365, 0.2241, -0.1294])
   h_1 = array([0.1294, 0.2241, -0.8365, 0.4830])

Reference: \"Orthonormal Bases of Compactly Supported Wavelets\",
            CPAM, Oct.89 
  """
  if (n % 2 != 0):
    raise Exception("No Daubechies filter exists for ODD length")
  k = n / 2
  a = p = q = 1
  h_0 = np.array([1, 1])
  for j in range(1, k):
    a = -a * 0.25 * (j + k - 1) / j
    h_0 = np.hstack((0, h_0)) + np.hstack((h_0, 0))
    p = np.hstack((0, -p)) + np.hstack((p, 0))
    p = np.hstack((0, -p)) + np.hstack((p, 0))
    q = np.hstack((0, q, 0)) + a*p
  q = np.sort(np.roots(q))
  qt = q[0:k-1]
  if (dtype == 'mid'):
    if (k % 2 == 1):
      qt = np.hstack((q[0:n-2:4], q[1:n-2:4]))
    else:
      qt = np.hstack((q[0], q[3:k-1:4], q[4:k-1:4], q[n-4:k:-4], q[n-5:k:-4]))
  h_0 = np.convolve(h_0, np.real(np.poly(qt)))
  h_0 = np.sqrt(2)*h_0 / sum(h_0)
  if (dtype == 'max'):
    h_0 = np.flipud(h_0)
  if (np.abs(sum(np.power(h_0, 2))) -1 > 1e-4):
    raise Exception("Numerically unstable for this value of n")
  h_1 = np.copy(np.flipud(h_0))
  h_1[0:n-1:2] = -h_1[0:n-1:2]
  return h_0, h_1

def hard_th(y, thld):
  """
HARDTH hard thresholds the input signal y with the threshold value
thld.

Input:  
   y    : 1D or 2D signal to be thresholded
   thld : threshold value

Output: 
   x : Hard thresholded output (x = (abs(y)>thld) * y)

Example:
   y = makesig('WernerSorrows', 8)
   thld = 1
   x = HardTh(y, thld)

Example Output:
  x = array([1.5545, 5.3175, 0, 1.6956, -1.2678, 0, 1.7332, 0])
  """
  return (np.abs(y) > thld) * y

def soft_th(y, thld):
  """
Soft thresholds the input signal y with the threshold value thld.

Input:  
   y    : 1D or 2D signal to be thresholded
   thld : Threshold value

Output: 
   x : Soft thresholded output (sign(y) * (x >= thld) * (x - thld))

Example:
   y = makesig('Doppler', 8)
   thld = 0.2
   x = soft_th(y, thld)

Example Output:
   x = array([0, 0, 0, -0.0703, 0, 0.2001, 0.0483, 0])

Reference: 
   \"De-noising via Soft-Thresholding\" Tech. Rept. Statistics,
   Stanford, 1992. D.L. Donoho.
  """
  x = np.abs(y)
  return np.sign(y) * (x >= thld) * (x - thld)

def makesig(signame, n = 512):
  """
Creates artificial test signal identical to the standard test 
signals proposed and used by D. Donoho and I. Johnstone in
WaveLab (- a matlab toolbox developed by Donoho et al. the statistics
department at Stanford University).

Input:  signame - Name of the desired signal
                    'HeaviSine'
                    'Bumps'
                    'Blocks'
                    'Doppler'
                    'Ramp'
                    'Cusp'
                    'Sing'
                    'HiSine'
                    'LoSine'
                    'LinChirp'
                    'TwoChirp'
                    'QuadChirp'
                    'MishMash'
                    'WernerSorrows' (Heisenberg)
                    'Leopold' (Kronecker)
        n       - Length in samples of the desired signal (Default 512)

Output: x   - resulting test signal

References:
        WaveLab can be accessed at
        www_url: http://playfair.stanford.edu/~wavelab/
        Also see various articles by D.L. Donoho et al. at
        web_url: http://playfair.stanford.edu/
  """
  t = np.array(range(1, n + 1)) / float(n)
  if (signame == 'HeaviSine'):
    y = 4 * np.sin(4 * np.pi * t)
    return y - np.sign(t - .3) - np.sign(.72 - t)
  if (signame == 'Bumps'):
    pos = np.array([.1, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81])
    hgt = np.array([4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
    wth = np.array([.005, .005, .006, .01, .01, .03, .01, .01, .005, .008, .005])
    y = np.zeros(n)
    for j in range(0, pos.size):
      y = y + hgt[j] / pow((1 + np.abs((t - pos[j]) / wth[j])), 4)
    return y
  if (signame == 'Blocks'):
    pos = np.array([.1, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81])
    hgt = np.array([4, (-5), 3, (-4), 5, (-4.2), 2.1, 4.3, (-3.1), 2.1, (-4.2)])
    y = np.zeros(n)
    for j in range(0, pos.size):
      y = y + (1 + np.sign(t - pos[j])) * (hgt[j]/2)
    return y
  if (signame == 'Doppler'):
    return np.sqrt(t * (1-t)) * np.sin((2 * np.pi * 1.05) / (t+.05))
  if (signame == 'Ramp'):
    return t - (t >= .37)
  if (signame == 'Cusp'):
    return np.sqrt(np.abs(t - .37))
  if (signame == 'Sing'):
    k = np.floor(n * .37)
    return 1 / np.abs(t - (k + .5)/n)
  if (signame == 'HiSine'):
    return np.sin(np.pi * (n * .6902) * t)
  if (signame == 'LoSine'):
    return np.sin(np.pi * (n * .3333) * t)
  if (signame == 'LinChirp'):
    return np.sin(np.pi * t * ((n * .125) * t))
  if (signame == 'TwoChirp'):
    return np.sin(np.pi * t * (n * t)) + np.sin((np.pi / 3) * t * (n * t))
  if (signame == 'QuadChirp'):
    return np.sin((np.pi/3) * t * (n * pow(t,2)))
  if (signame == 'MishMash'):
    y = np.sin((np.pi/3) * t * (n * pow(t,2)))
    y = y + np.sin(np.pi * (n * .6902) * t)
    return y + np.sin(np.pi * t * (n * .125 * t))
  if (signame == 'WernerSorrows'):
    y = np.sin(np.pi * t * (n/2 * pow(t, 2)))
    y = y + np.sin(np.pi * (n * .6902) * t)
    y = y + np.sin(np.pi * t * (n * t))
    pos = np.array([.1, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81])
    hgt = np.array([4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
    wth = np.array([.005, .005, .006, .01, .01, .03, .01, .01, .005, .008, .005])
    for j in range(0, pos.size):
      y = y + hgt[j] / pow((1 + np.abs((t - pos[j]) / wth[j])), 4)
    return y
  if (signame == 'Leopold'):
    return (t == np.floor(.37 * n)/n) * 1.0

def denoise(x, h, denoise_type = 0, option = None):
  """
DENOISE is a generic routine for wavelet based denoising.
The routine will denoise the signal x using the 2-band wavelet
system described by the filter h using either the traditional 
discrete wavelet transform (DWT) or the linear shift invariant 
discrete wavelet transform (also known as the undecimated DWT
(UDWT)). 

Input:  
   x            : 1D or 2D signal to be denoised
   h            : Scaling filter to be applied
   denoise_type : Type of transform (Default: type = 0)
                  0 --> Discrete wavelet transform (DWT)
                  1 --> Undecimated DWT (UDWT)
   option       : Default settings is marked with '*':
                  *type = 0 --> option = [0 3.0 0 0 0 0]
                  type = 1 --> option = [0 3.6 0 1 0 0]
   option(1)    : Whether to threshold low-pass part
                  0 --> Don't threshold low pass component 
                  1 --> Threshold low pass component
   option(2)    : Threshold multiplier, c. The threshold is
                  computed as: 
                    thld = c*MAD(noise_estimate)). 
                  The default values are:
                    c = 3.0 for the DWT based denoising
                    c = 3.6 for the UDWT based denoising
   option(3)    : Type of variance estimator
                  0 --> MAD (mean absolute deviation)
                  1 --> STD (classical numerical std estimate)
   option(4)    : Type of thresholding
                  2 --> Soft thresholding
                  1 --> Hard thresholding
   option(5)    : Number of levels, L, in wavelet decomposition. By
                  setting this to the default value '0' a maximal
                  decomposition is used.
   option(6)    : Actual threshold to use (setting this to
                  anything but 0 will mean that option(3)
                  is ignored)

Output: 
   xd           : Estimate of noise free signal 
   xn           : The estimated noise signal (x-xd)
   option       : A vector of actual parameters used by the
                  routine. The vector is configured the same way as
                  the input option vector with one added element
                  option(7) = type.

Example 1: 
   from numpy.random import randn
   N = 16
   h = daubcqf(6)[0]
   s = makesig('Doppler', N)
   n = randn(1,N)
   x = s + n/10 # (approximately 10dB SNR)
   %Denoise x with the default method based on the DWT
   xd, xn, opt1 = denoise(x,h)
   %Denoise x using the undecimated (LSI) wavelet transform
   yd, yn, opt2 = denoise(x,h,1)

Example 2: (on an image)  
   from scipy.io import loadmat
   from numpy.random import random_sample
   lena = loadmat('../tests/lena512.mat')['lena512']
   h = daubcqf(6)[0]
   noisyLena = lena + 25 * random_sample(lena.shape)
   denoisedLena, xn, opt1 = denoise(noisyLena, h)
  """
  if (option == None and denoise_type == 0):
    option = [0, 3.0, 0, 2, 0, 0]
  if (option == None and denoise_type == 1):
    option = [0, 3.6, 0, 1, 0, 0]
  if (type(option) != list):
    option = list(option)
  mx = x.shape[0]
  nx = 1
  if (len(x.shape) > 1):
    nx = x.shape[1]
  dim = min(mx, nx)
  n = dim
  if (dim == 1):
    n = max(mx, nx)
  if (option[4] == 0):
    L = np.int(np.floor(np.log2(n)))
  else:
    L = option[4]
  if (denoise_type == 0):
    xd = dwt(x, h, L)[0]
    if (option[5] == 0):
      if (nx > 1):
        tmp = xd[np.floor(mx/2):mx, np.floor(nx/2):nx]
      else:
        tmp = xd[np.floor(mx/2):mx]
      if (option[2] == 0):
        thld = option[1] * np.median(np.abs(tmp)) / .67
      elif (option[2] == 1):
        thld = option[1] * np.std(tmp, ddof=1)
    else:
      thld = option[5]
    if (dim == 1):
      ix = np.array(range(0, (n/(np.power(2, L)))))
      if (ix.size == 1):
        ix = ix[0]
      ykeep = xd[ix]
    else:
      ix = np.array(range(0, (mx/(np.power(2,L)))))
      jx = np.array(range(0, (nx/(np.power(2,L)))))
      if (ix.size == 1):
        ix = ix[0]
      if (jx.size == 1):
        jx = jx[0]
      ykeep = xd[ix, jx]
    if (option[3] == 2):
      xd = soft_th(xd, thld)
    elif (option[3] == 1):
      xd = hard_th(xd, thld)
    if (option[0] == 0):
      if (dim == 1):
        xd[ix] = ykeep
      else:
        xd[ix, jx] = ykeep
    xd = idwt(xd, h, L)[0]
  elif (denoise_type == 1):
    (xl, xh, L) = rdwt(x, h, L)
    easter_egg = 23
    if (dim == 1):
      c_offset = 0
    else:
      c_offset = 2 * nx
    if (option[5] == 0):
      if (nx > 1):
        tmp = xh[:,c_offset:c_offset+mx] 
      else:
        tmp = xh[c_offset:c_offset+mx:1] 
      if (option[2] == 0):
        thld = option[1] * np.median(np.abs(tmp)) / .67
      elif (option[2] == 1):
        thld = option[1] * np.std(tmp, ddof=1)
    else:
      thld = option[5]
    if (option[3] == 2):
      xh = soft_th(xh, thld)
      if (option[0] == 1):
        xl = soft_th(xl, thld)
    elif (option[3] == 1):
      xh = hard_th(xh, thld)
      if (option[0] == 1):
        xl = hard_th(xl, thld)
    xd = irdwt(xl, xh, h, L)[0]
  option[5] = (thld)
  option.append(denoise_type)
  xn = x - xd
  return xd, xn, option

%}
