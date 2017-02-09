
BLS-GSM Image Denoising Matlab Toolbox 1.0.3
============================================ 		 14 May 2004 
					Latest Revision: 23 Feb 2005


Thank you for using this software. 

BLS-GSM stands for "Bayesian Least Squares - Gaussian Scale Mixture". 

This toolbox implements the algorithm described in: 

J Portilla, V Strela, M Wainwright, E P Simoncelli, "Image Denoising
using Scale Mixtures of Gaussians in the Wavelet Domain," IEEE
Transactions on Image Processing. vol 12, no. 11, pp. 1338-1351,
November 2003 

adding some more possibilities for representations, besides the full
steerable pyramid, as explained in the package description below. This
tool implements image denoising assuming that the noise is Gaussian,
and that we know its power spectral density (it does not need to be white).
I have written it using as a starting seed some block-Wiener denoising
Matlab code written by Vasily Strela back in the year 2000. 

The package also contains some previously released public domain
software authored by Eero P. Simoncelli, belonging to his MatlabPyrTools
toolbox (available at http://www.cns.nyu.edu/~eero/software.html), and
also some public domain routines taken from Rice Wavelet Toolbox
(http://www.dsp.rice.edu/software/rwt.shtml). 



INSTALLATION ============ 

1) Use MATLAB@ 6.0 version or posterior (it has not been tested with
older versions)

2) Unpack the zip file into a directory in your hard drive, respecting
the original tree of subdirectories included in the pack 

3) Make available this directory and all subdirectories to MATLAB@
by including the parent directory in the MATLAB@ path and checking the
appropriated box for including the subdirectories in the path 

After the installation, you should be able to run the demo program
("denoi_demo"). 



PACKAGE DESCRIPTION =================== 

MAIN DIRECTORY 

* Demo program ("denoi_demo.m", in the main directory) 

It shows an example of aplication of the method to a simulated noisy
image. The idea is that the parameters can be changed by the user,
experimenting their influence. This applies especially to the noise
parameters (variance and power spectral density) and the representation
parameters (explained below). It is also possible to modify the model
parameters (size of the neighborhood, including a parent or not, etc.),
although we recommend not to change them (except for, possibly, the
inclusion or not of a "parent" in the neighborhood).

Please, note that, by default, the code uses representation parameters
that do NOT correspond to the representation used in our work IEEE TIP
Nov 2003. The default set of parameters is faster, but does not always
provide the optimal results. For reproducing the results of the referred
article, please uncomment the appropriated lines in the "denoi_demo.m"
program.

* ReadMe.txt (this file) 

DENOISING_SUBPROGRAMS DIRECTORY 

* Main function for noise removal ("denoi_BLS_GSM.m") 

This function takes a noisy image, and the (noise, representation and
model) parameters and applies the referred algorithm for cleaning it.
Besides dealing with issues such as boundary handling (mirror reflection
to avoid boundary artifacts), its main task is performed by a call to
the routines "decomp_recons_X.m", which are described below. 

* Converting from the image domain to the wavelet domain and vice versa
("decomp_reconst_X.m") 

(The "X" in "denoi_reconst_X.m" represents several options that are
described below). These functions take the noisy image (possibly
extended for boundary handling purposes) and apply the chosen
representation to transform it into the wavelet domain. Once it has been
decomposed into subbands, the latter are processed sequentially (in
couples corresponding to subbands of the same orientations adjacent in
the scale, in case of the model parameter "parent" has been set to "1",
or individually, otherwise), and then recomposed into a single image. In
the current version, 4 main representation options are available
(corresponding to the representation parameter "repres1"): 

- Option "w": Orthogonal wavelet. Very fast, but of relatively poor
denoising performance, because of not being translation invariant. 

- Option "uw": Undecimated version of orthogonal wavelet. In fact, this
is a highly redundant version of the orthogonal wavelet that avoids
intra-subband aliasing, but still keeps a multiscale structure. (The
lowest level of the pyramid are extended 2 in each dimension, whereas
the rest of the scales are extended 4 times in each dimension.) It
provides a good trade-off between computational cost and denoising
performance (which is very high). 

- Option "s": Steerable Pyramid (Simoncelli & Freeman, 1995). It allows
the user to choose an arbitrary number of orientations. The splitting in
oriented subbands is not applied to the very high frequencies, which are
represented in a single subband (high-pass residual). With a moderate
computational cost for a modest number of orientations (4 or less), its
results depend on the type of image, being comparable (or slightly
worse) on average than those obtained with option "uw". 

- Option "fs": Full steerable pyramid (described in the article cited
above). Same as previous one, but now also the very high frequencies are
splitted into orientations. It provides very high denoising performance
(for some images slightly better than "uw"), especially with high Nor
(for instance, 8), but it is very demanding computationally. 

The relative effect on the denoising performance of using each of these
representations is evaluated in the referred IEEE TIP article. For the
"w" and "uw" representation options, there is also the option of
choosing a particular wavelet. We recommend the use of Daubechies
filters of order from 1 (Haar) to 3 or 4. For more information, please
read the help information included within the functions (or type "help
<function_name>"). 

* Subband Noise removal function ("denoi_BLS_GSM_band.m") 

This function is called by "decomp_reconst_X.m" and implements the
kernel of the algorithm. It is only one, independently from which
representation is applied. 

SIMONCELLI_PYR_TOOLS DIRECTORY 

A subset with the required functions from Prof. Eero Simoncelli's
"MatlabPyrTools" toolbox (available at
http://www.cns.nyu.edu/~eero/software.html ). It includes the functions
that implement the steerable pyramid decomposition and reconstruction,
and some other functions used for the manipulation of the subbands, in
both its standard and its full version (full steerable pyramid, for the
"fs" option). 

ADDED_PYR_TOOLS DIRECTORY 

Functions that implement additional utilities, like the DWT code used
for building aliasing-free multi-scale transforms (for the "uw" option),
which was taken from the Rice Wavelet Toolbox
(http://www.dsp.rice.edu/software/rwt.shtml). Some others ({\it
expand.m, shrink.m, snr.m}) are general purpose functions written by
me (to expand and shrink an image in the Fourier domain and to compute
the signal-to-noise ratio in dB, respectively). 

IMAGES 

Several images in PNG (lossless compressed) format, to test the method.
They include the ones used in the IEEE TIP referred article. 



Please, report to javier@decsai.ugr.es any bug or comment. 

Javier Portilla 
http://decsai.ugr.es/~javier

