#include "mex.h"
#include "math.h"
#include "string.h"
/*   undef needed for LCC compiler  */
#undef EXTERN_C
/* Multi-threading libraries */
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#endif

__inline double pow2(double a) { return a*a; }

double * gaussian_kernel_2D(int kernelratio){
    /* Create the Gaussian 2D kernel */
    int kernelsize, npixels, x, y, i, j, p;
    double sumK, sigma, *K;
    kernelsize=kernelratio*2+1;
    sigma=((double)kernelsize)/4.0; p=0;
    npixels=kernelsize*kernelsize;
    K=(double*)malloc(npixels*sizeof(double));
    for (i=0; i<kernelsize; i++) {
        for (j=0; j<kernelsize; j++) {
            x=i-kernelratio; y=j-kernelratio;
            K[p] = exp(-((pow2(x)+pow2(y))/(2.0*pow2(sigma))));
            p++;
        }
    }
    /* Normalize kernel */
    sumK=0; for(i=0; i<npixels; i++) { sumK+=K[i]; }
    for(i=0; i<npixels; i++) { K[i]/=(sumK+1e-15); }
    return K;
}

double * gaussian_kernel_3D(int kernelratio) {
    /* Create the Gaussian 3D kernel */
    int kernelsize, npixels, x, y, z, i, j, k, p;
    double sumK, sigma, *K;
    kernelsize=kernelratio*2+1; sigma=((double)kernelsize)/4.0; p=0;
    npixels=kernelsize*kernelsize*kernelsize;
    K=(double*)malloc(npixels*sizeof(double));
    for (i=0; i<kernelsize; i++) {
        for (j=0; j<kernelsize; j++) {
            for (k=0; k<kernelsize; k++) {
                x=i-kernelratio; y=j-kernelratio; z=k-kernelratio;
                K[p] = exp(-((pow2(x)+pow2(y)+pow2(z))/(2.0*pow2(sigma)))); p++;
            }
        }
    }
    /* Normalize kernel */
    sumK=0; for(i=0; i<npixels; i++) { sumK+=K[i]; }
    for(i=0; i<npixels; i++) { K[i]/=(sumK+1e-15); }
    return K;
}

void get2Dvectors(double *I, int *Isize, double *V, int *Vsize, int kernelratio, int *block, double *K, int ThreadID, int Nthreads) {
    int indexI, indexI_part1, indexI_part2, indexI_part3;
    int indexV=0;
    int kernelsize;
    int npixels2;
    int C1;
    int x, y, p;
    int tz, ik, jk;
    int block_size[2];

    block_size[0]=block[2]-block[0]+1;
    block_size[1]=block[3]-block[1]+1;
    kernelsize=2*kernelratio+1;
    C1=-(block[0]+block[1]*block_size[0])*Vsize[0];
    npixels2=Isize[0]*Isize[1];
    /* Loop through the block */
    for(y=block[1]; y<=block[3]; y++) {
        for(x=block[0]+ThreadID; x<=block[2]; x+=Nthreads) {
            indexV=(x+y*block_size[0])*Vsize[0]+C1;
    
            /* Get a patch*/
            indexI_part1=0;
            for(tz=0; tz<Isize[2]; tz++) {
                p=0;
                indexI_part3=(y-kernelratio)*Isize[0];
                indexI_part2=x-kernelratio+indexI_part1;
                indexI=indexI_part2+ indexI_part3;
                for (jk=0; jk<kernelsize; jk++) {
                    for (ik=0; ik<kernelsize; ik++) {
                        V[indexV]=K[p]*I[indexI];
                        indexV++; indexI++;
                        p++;
                    }
                    indexI_part3+= Isize[0];
                    indexI=indexI_part2+indexI_part3;
                }
                indexI_part1+=npixels2;
            }
        }
    }
}

void get3Dvectors(double *I, int *Isize, double *V, int *Vsize, int kernelratio, int *block, double *K, int ThreadID, int Nthreads) {
    int indexI, indexIpart1, indexIpart2, indexIpart3, indexIpart4, indexIpart5, indexIpart6;
    int indexV=0;
    int kernelsize;
    int npixels2;
    int x, y, z, p;
    int ik, jk, kk;
    int C1, C2;
    int block_size[3];
    
    indexV+=ThreadID*Vsize[0];
    kernelsize=2*kernelratio+1;
    npixels2=Isize[0]*Isize[1];
    indexIpart4=-kernelratio-kernelratio*Isize[0]-kernelratio*npixels2;
    indexIpart5=block[2]*npixels2;
    indexIpart6=block[1]*Isize[0];
    block_size[0]=block[3]-block[0]+1;
    block_size[1]=block[4]-block[1]+1;
    block_size[2]=block[5]-block[2]+1;
    C2=block_size[0]*block_size[1];
    C1=-(block[0]+block[1]*block_size[0]+block[2]*C2)*Vsize[0];

    /* Loop through the block */
    for(z=block[2]; z<=block[5]; z++) {
        indexIpart3=indexIpart6;
        for(y=block[1]; y<=block[4]; y++) {
            for(x=block[0]+ThreadID; x<=block[3]; x+=Nthreads) {
                indexV=(x+y*block_size[0]+z*C2)*Vsize[0]+C1;
                /* Get a patch*/
                p=0;
                indexIpart1=indexIpart5+x+indexIpart4;
                for(kk=-kernelratio; kk<=kernelratio; kk++) {
                    indexIpart2=indexIpart3+indexIpart1;
                    for (jk=-kernelratio; jk<=kernelratio; jk++) {
                        indexI=indexIpart2;
                        for (ik=-kernelratio; ik<=kernelratio; ik++) {
                            V[indexV]=K[p]*I[indexI];
                            indexV++;
                            indexI++;
                            p++;
                        }
                        indexIpart2+=Isize[0];
                    }
                    indexIpart1+=npixels2;
                }
                indexV+=(Nthreads-1)*Vsize[0];
            }
            indexIpart3+=Isize[0];
        }
        indexIpart5+=npixels2;
        
    }
}

#ifdef _WIN32
 unsigned __stdcall getvectors_multi_threaded(double **Args){
#else
 void getvectors_multi_threaded(double **Args){
#endif
    
    /* Input image, output vectors */
    double *I, *V;
    double *K;
    /* Size of input image */
    int Isize[3];
    /* Size of input vectors */
    int Vsize[3];
    /* Size of vector volume */
    int image3D;
    /* Constants used */
    int kernelratio;
    int block[6];
    
    int ThreadID;
    int Nthreads;
    
    I=Args[0];
    Isize[0]=(int)Args[1][0];
    Isize[1]=(int)Args[1][1]; 
    Isize[2]=(int)Args[1][2];
    V=Args[2];
    Vsize[0]=(int)Args[3][0];
    Vsize[1]=(int)Args[3][1]; 
    Vsize[2]=(int)Args[3][2];
    kernelratio=(int)Args[4][0];
    image3D=(int)Args[4][1];
    block[0]=(int)Args[5][0]; 
    block[1]=(int)Args[5][1];
    block[2]=(int)Args[5][2]; 
    block[3]=(int)Args[5][3];
    block[4]=(int)Args[5][4]; 
    block[5]=(int)Args[5][5];
    K=Args[6];
    ThreadID=(int)Args[7][0];
    Nthreads=(int)Args[8][0];
    
    if(image3D==0) {
        get2Dvectors(I, Isize, V, Vsize, kernelratio, block, K, ThreadID, Nthreads);
    }
    else {
        get3Dvectors(I, Isize, V, Vsize, kernelratio, block, K, ThreadID, Nthreads);
    }
    
    /*  explicit end thread, helps to ensure proper recovery of resources allocated for the thread */
    #ifdef _WIN32
            _endthreadex( 0 );
    return 0;
    #else
            pthread_exit(NULL);
    #endif
            
}

/* The matlab mex function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {
    /* Input image, output image */
    double *I, *V;
    
    double *K;
    
    /* Size of input image */
    int Isize[3]={1, 1, 1};
    const mwSize *dimsI;
    int ndimsI;
    
    /* Size of vector volume */
    int nVsize=2;
    int Vsize[3];
    int indexV=0;
    
    /* Constants used */
    int kernelratio=3;
    int kernelsize;
    
    int block[6]={1, 1, 1, 1, 1, 1};
    int block_size[3];
    
    int image3D;
    
    /* Loop variable */
    int i;
    double *T;
    double Isize_d[3];
    double Vsize_d[3];
    double par_d[4];
    double block_d[6];

    double *Nthreadsd;
    int Nthreads;
    
    
    /* double pointer array to store all needed function variables) */
    double ***ThreadArgs;
    double **ThreadArgs1;
    
    /* Handles to the worker threads */
    #ifdef _WIN32
            HANDLE *ThreadList;
    #else
            pthread_t *ThreadList;
    #endif
            
            /* ID of Threads */
            double **ThreadID;
    double *ThreadID1;
    
    
    /* Check for proper number of arguments. */
    if(nrhs<3) {
        mexErrMsgTxt("At least three inputs required.");
    } else if(nlhs!=1) {
        mexErrMsgTxt("One output required");
    }
    
    /* Check if all values are inputs are of type double*/
    for(i=0; i<nrhs; i++) {
        if(!mxIsDouble(prhs[i])) { mexErrMsgTxt("Inputs must be double"); }
    }
    
    /* Check input image dimensions */
    ndimsI=mxGetNumberOfDimensions(prhs[0]);
    if((ndimsI<2)||(ndimsI>3)) { mexErrMsgTxt("Image must be 2D"); }
    dimsI= mxGetDimensions(prhs[0]);
    Isize[0]=dimsI[0]; Isize[1]=dimsI[1];
    if(ndimsI==3) { Isize[2]=dimsI[2]; }
    
    if(Isize[2]>3) { image3D=1; } else { image3D=0; }
    
    /* Connect input image */
    I=(double *)mxGetData(prhs[0]);
    
    /* Set Values */
    T=(double *)mxGetData(prhs[1]);

    kernelratio=(int)T[0];
    kernelsize=2*kernelratio+1;
    
    Nthreadsd=(double *)mxGetData(prhs[2]);
    
    
    if(image3D==0) {
        block[0]=kernelratio;
        block[1]=kernelratio;
        block[2]=Isize[0]-kernelratio-1;
        block[3]=Isize[1]-kernelratio-1;
        
        block_size[0]=block[2]-block[0]+1;
        block_size[1]=block[3]-block[1]+1;
        
        Vsize[0]=kernelsize*kernelsize*Isize[2];
        Vsize[1]=block_size[0]*block_size[1];
    }
    else {
        block[0]=kernelratio;
        block[1]=kernelratio;
        block[2]=kernelratio;
        block[3]=Isize[0]-kernelratio-1;
        block[4]=Isize[1]-kernelratio-1;
        block[5]=Isize[2]-kernelratio-1;
        
        block_size[0]=block[3]-block[0]+1;
        block_size[1]=block[4]-block[1]+1;
        block_size[2]=block[5]-block[2]+1;
        
        Vsize[0]=kernelsize*kernelsize*kernelsize;
        Vsize[1]=block_size[0]*block_size[1]*block_size[2];
    }
    
    /* Create output array */
    plhs[0] = mxCreateNumericArray(nVsize, Vsize, mxDOUBLE_CLASS, mxREAL);
    V=(double *)mxGetData(plhs[0]);
    
    if(image3D==0) {
        K = gaussian_kernel_2D(kernelratio);
        for (i=0; i<(kernelsize*kernelsize); i++)  { K[i]=sqrt(K[i]); }

    }
    else {
        K = gaussian_kernel_3D(kernelratio);
        for (i=0; i<(kernelsize*kernelsize*kernelsize); i++)  { K[i]=sqrt(K[i]); }
    
    }
    
    
    Nthreads=(int)Nthreadsd[0];
    
    /* Reserve room for handles of threads in ThreadList  */
    #ifdef _WIN32
            ThreadList = (HANDLE*)malloc(Nthreads* sizeof( HANDLE ));
    #else
            ThreadList = (pthread_t*)malloc(Nthreads* sizeof( pthread_t ));
    #endif
            
    ThreadID = (double **)malloc( Nthreads* sizeof(double *) );
    ThreadArgs = (double ***)malloc( Nthreads* sizeof(double **) );
    
    
    for(i=0; i<3; i++) {
        Isize_d[i]=(double)Isize[i];
        Vsize_d[i]=(double)Vsize[i];
        block_d[i] =(double)block[i];
        block_d[i+3] =(double)block[i+3];
    }
    par_d[0] =(double)kernelratio;
    par_d[1] =(double)image3D;
            
    for (i=0; i<Nthreads; i++) {
        /*  Make Thread ID  */
        ThreadID1= (double *)malloc( 1* sizeof(double) );
        ThreadID1[0]=i;
        ThreadID[i]=ThreadID1;
        
        /*  Make Thread Structure  */
        ThreadArgs1 = (double **)malloc( 9* sizeof( double * ) );
       
        ThreadArgs1[0]=I;
        ThreadArgs1[1]=Isize_d;
        ThreadArgs1[2]=V;
        ThreadArgs1[3]=Vsize_d;
        ThreadArgs1[4]=par_d;
        ThreadArgs1[5]=block_d;
        ThreadArgs1[6]=K;
        ThreadArgs1[7]=ThreadID[i];
        ThreadArgs1[8]=Nthreadsd;
        
        /* Start a Thread  */
        ThreadArgs[i]=ThreadArgs1;
                
        #ifdef _WIN32
                ThreadList[i] = (HANDLE)_beginthreadex( NULL, 0, &getvectors_multi_threaded, ThreadArgs[i] , 0, NULL );
        #else
                pthread_create((pthread_t*)&ThreadList[i], NULL, (void *) &getvectors_multi_threaded, ThreadArgs[i]);
        #endif
    }
    
    #ifdef _WIN32
        for (i=0; i<Nthreads; i++) { WaitForSingleObject(ThreadList[i], INFINITE); }
        for (i=0; i<Nthreads; i++) { CloseHandle( ThreadList[i] ); }
    #else
        for (i=0; i<Nthreads; i++) { pthread_join(ThreadList[i], NULL); }
    #endif
            
            
    for (i=0; i<Nthreads; i++) {
        free(ThreadArgs[i]);
        free(ThreadID[i]);
    }
    
    free(ThreadArgs);
    free(ThreadID );
    free(ThreadList);
    free(K);
}



