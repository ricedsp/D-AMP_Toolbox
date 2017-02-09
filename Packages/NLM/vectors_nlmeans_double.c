#include "mex.h"
#include "math.h"
#include "string.h"
#ifndef min
#define min(a, b)        ((a) < (b) ? (a): (b))
#endif
#ifndef max
#define max(a, b)        ((a) > (b) ? (a): (b))
#endif
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

void filter2D(double *I, int *Isize, double *J, int *dimsJ, double *V, int *Vsize, int kernelratio, int windowratio, double filterstrength, int *block, int *block_size, int ThreadID, int Nthreads) {
    int indexV1, indexV2, indexV1part1, indexV2part1;
    int indexI;
    int indexJ;
    double w;
    double average[3];
    double wmax;
    double sweight;
    double distance;
    int npixels2;
    int windowsize;
    double filterstrength2;
    int kernelsize;
    int block_kernel[6]={1, 1, 1, 1, 1, 1};
    int block_kernel_sizex;
    int block_kernel_sizey;
    int i, x, y;
    int ik, jk;
    int xv1, yv1, xv2, yv2;

    kernelsize=2*kernelratio+1;
    windowsize=2*windowratio+1;
    
    /* Calculate block size */
    block_kernel[0]=kernelratio;
    block_kernel[1]=kernelratio;
    block_kernel[2]=Isize[0]-kernelratio-1;
    block_kernel[3]=Isize[1]-kernelratio-1;
    
    block_kernel_sizex=block_kernel[2]-block_kernel[0]+1;
    block_kernel_sizey=block_kernel[3]-block_kernel[1]+1;
    
    npixels2=Isize[0]*Isize[1];
    filterstrength2=1/pow2(filterstrength);
    
    /* Loop through the block */
    for(y=block[1]; y<=block[3]; y++) {
        yv1=y-kernelratio;
        indexV1part1=yv1*block_kernel_sizex;
        for(x=block[0]+ThreadID; x<=block[2]; x+=Nthreads) {
            average[0]=0; average[1]=0; average[2]=0;
            wmax=0;
            sweight=0;
            
            /* Calculate Vector index */
            xv1=x-kernelratio;
            indexV1=(xv1+indexV1part1)*Vsize[0];
            
            /* Loop through the search window */
            for (jk=-windowratio; jk<=windowratio; jk++) {
                yv2=yv1+jk;
                indexV2part1=yv2*block_kernel_sizex;
                for (ik=-windowratio; ik<=windowratio; ik++) {
                    if((ik==0)&&(jk==0)) { continue;  }
                    
                    /* Calculate Vector index */
                    xv2=xv1+ik;
                    indexV2=(xv2+indexV2part1)*Vsize[0];
                    distance=0;
                    for(i=0; i<Vsize[0]; i++) { distance+=pow2(V[indexV1+i]-V[indexV2+i]); }
                    w=exp(-distance*filterstrength2);
                    wmax=max(w, wmax);
                    
                    sweight+=w;
                    indexI=(x+ik)+(y+jk)*Isize[0];
                    for(i=0; i<Isize[2]; i++) { average[i]+=w*I[indexI];indexI+=npixels2; }
                }
            }
            /* At the last pixel */
            wmax=max(wmax, 1e-15);
            sweight=sweight+wmax;
            indexI=x+y*Isize[0];
            for(i=0; i<Isize[2]; i++) {
                average[i]+=wmax*I[indexI]; indexI+=npixels2;
                average[i]/=sweight;
            }
            
            /* Set the filterd pixel */
            indexJ=(x-block[0])+(y-block[1])*block_size[0];
            for(i=0; i<Isize[2]; i++) {
                J[indexJ]=average[i]; indexJ+=block_size[0]*block_size[1];
            }
        }
    }
}


void Filterstep3D(int *P, int windowratio, double filterstrength2, double *J, int indexJ, int indexV1, int *Vsize, double *V, int *Isize, double *I) {
    double w;
    double average;
    double wmax;
    double sweight;
    double distance;
    int ik, jk, kk;
    int indexV2;
    int i;
    int indexI;
    average=0; wmax=0; sweight=0;
    /* Loop through the search window */
    for (kk=-windowratio; kk<=windowratio; kk++) {
        P[1]=P[8]+P[9]+P[5]; P[16]=P[20]+P[15];
        for (jk=-windowratio; jk<=windowratio; jk++) {
            indexV2=P[1];
            indexI=P[16];
            for (ik=-windowratio; ik<=windowratio; ik++) {
                if((ik!=0)||(jk!=0)||(kk!=0)) {
                    /* Calculate Vector index */
                    distance=0;
                    for(i=0; i<Vsize[0]; i++)  { distance+=pow2(V[indexV1+i]-V[indexV2+i]); }
                    w=exp(-distance*filterstrength2); wmax=max(w, wmax); sweight+=w;
                    average+=w*I[indexI];
                }
                indexI++;
                indexV2+=Vsize[0];
            }
            P[1]+=P[6]; P[16]+=Isize[0];
        }
        P[9]+=P[0]; P[15]+=P[12];
    }
    /* At the lasst pixel */
    wmax=max(wmax, 1e-15);
    sweight+=wmax;
    average+=wmax*I[ P[13]]; average/=sweight;
    
    /* Set the filterd pixel */
    J[indexJ]=average;
}



void filter3D(double *I, int *Isize, double *J, int *dimsJ, double *V, int *Vsize, int kernelratio, int windowratio, double filterstrength, int *block, int *block_size, int ThreadID, int Nthreads) {
    int indexV1;
    int indexJ;
    int windowsize;
    double filterstrength2;
    int kernelsize;
    int block_kernel[6]={1, 1, 1, 1, 1, 1};
    int block_kernel_sizex;
    int block_kernel_sizey;
    int block_kernel_sizez;
    int x, y, z;
    int P[50];
    kernelsize=2*kernelratio+1;
    windowsize=2*windowratio+1;
    
    /* Calculate block size */
    block_kernel[0]=kernelratio;
    block_kernel[1]=kernelratio;
    block_kernel[2]=kernelratio;
    block_kernel[3]=Isize[0]-kernelratio-1;
    block_kernel[4]=Isize[1]-kernelratio-1;
    block_kernel[5]=Isize[2]-kernelratio-1;
    
    block_kernel_sizex=block_kernel[3]-block_kernel[0]+1;
    block_kernel_sizey=block_kernel[4]-block_kernel[1]+1;
    block_kernel_sizez=block_kernel[5]-block_kernel[2]+1;
    
    filterstrength2=1/pow2(filterstrength);
    P[0]=block_kernel_sizex*block_kernel_sizey*Vsize[0];
    P[3]=-windowratio*Vsize[0];
    P[6]=block_kernel_sizex*Vsize[0];
    P[7]=-windowratio*P[6];
    P[11]=windowratio*P[0];
    P[12]=Isize[1]*Isize[0];
    P[14]=block[0]+block[1]*block_size[0]+block[2]*block_size[0]*block_size[1];
    P[17]=block_kernel_sizex*block_kernel_sizey;
    P[20]=-windowratio*Isize[0]-windowratio;
    P[21]=(block[0]-kernelratio)*Vsize[0]+P[3];
    P[22]=block_size[0]*block_size[1];
    P[27]=-windowratio*P[12];
    P[28]=block_kernel_sizex*Vsize[0];
    P[29]=block[1]*Isize[0];
    P[30]=block[1]*block_size[0];
    P[31]=block[1]*P[6]-kernelratio*P[6]+P[7];
    P[18]=(block[2]-kernelratio)*P[17]*Vsize[0];
    P[10]=(block[2]-kernelratio)*P[0]-P[11];
    P[23]=block[2]*P[22]-P[14];
    P[25]=block[2]*P[12];
    P[32]=block[1]*P[28]-kernelratio*P[28];
    P[33]=(block[0]-kernelratio)*Vsize[0];
    /* Loop through the block */
    P[44]=ThreadID*Vsize[0];
    P[45]=Nthreads*Vsize[0];
    
    
    for(z=block[2]; z<=block[5]; z++) {
        P[24]=P[30]+P[23];
        P[26]=P[29]+P[25];
        P[8] =P[31];
        P[19]=P[32]+P[18];
        for(y=block[1]; y<=block[4]; y++) {
            P[5]=P[21]; P[13]=block[0]+P[26];
            
            indexV1=P[19]+P[33];
            indexJ=block[0]+P[24];
            
            indexV1+=P[44];
            indexJ+=ThreadID;
            P[13]+=ThreadID; 
            P[5]+=P[44];
                
            for(x=block[0]+ThreadID; x<=block[3]; x+=Nthreads) {
                P[9]=P[10]; P[15]=P[27]+P[13];
                
                Filterstep3D(P, windowratio, filterstrength2, J, indexJ, indexV1, Vsize, V, Isize, I);
                
                indexV1+=P[45];
                indexJ+=Nthreads;
                P[13]+=Nthreads; 
                P[5]+=P[45];
            }
            P[24]+=block_size[0];P[26]+=Isize[0]; P[8]+=P[6]; P[19]+=P[28];
        }
        
        P[18]+=P[17]*Vsize[0];
        P[10]+=P[0];
        P[23]+=P[22];
        P[25]+=P[12];
    }
    
}


#ifdef _WIN32
 unsigned __stdcall filter_multi_threaded(double **Args){
#else
 void filter_multi_threaded(double **Args){
#endif
    /* Input image, output image */
    double *I, *J, *V;
    /* Size of input image */
    int Isize[3];
    /* Size of input vectors */
    int Vsize[3];
    /* Size of vector volume */
    int dimsJ[3];
    int image3D;
    /* Constants used */
    int windowratio;
    double filterstrength;
    int kernelratio=3;
    int block[6];
    int block_size[3];
    
    int ThreadID;
    int Nthreads;
    
    Isize[0]=(int)Args[0][0]; Isize[1]=(int)Args[0][1]; Isize[2]=(int)Args[0][2];
    dimsJ[0]=(int)Args[1][0]; dimsJ[1]=(int)Args[1][1]; dimsJ[2]=(int)Args[1][2];
    Vsize[0]=(int)Args[2][0]; Vsize[1]=(int)Args[2][1]; Vsize[2]=(int)Args[2][2];
    block_size[0]=(int)Args[3][0]; block_size[1]=(int)Args[3][1]; block_size[2]=(int)Args[3][2];
    
    kernelratio=(int)Args[4][0]; windowratio=(int)Args[4][1];
    filterstrength=Args[4][2]; image3D=(int)Args[4][3];
    
    block[0]=(int)Args[5][0]; block[1]=(int)Args[5][1];
    block[2]=(int)Args[5][2]; block[3]=(int)Args[5][3];
    block[4]=(int)Args[5][4]; block[5]=(int)Args[5][5];
    
    I=Args[6];
    J=Args[7];
    V=Args[8];
    ThreadID=(int)Args[9][0];
    Nthreads=(int)Args[10][0];
    
    if(image3D==0) {
        filter2D(I, Isize, J, dimsJ, V, Vsize, kernelratio, windowratio, filterstrength, block, block_size, ThreadID, Nthreads);
    }
    else {
        filter3D(I, Isize, J, dimsJ, V, Vsize, kernelratio, windowratio, filterstrength, block, block_size, ThreadID, Nthreads);
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
    double *I, *J, *V;
    
    /* Size of input image */
    int Isize[3]={1, 1, 1};
    const mwSize *dimsI;
    int ndimsI;
    
    /* Size of input vectors */
    int Vsize[2]={1, 1};
    const mwSize *dimsV;
    int ndimsV;
    
    /* Size of vector volume */
    int ndimsJ=3;
    int dimsJ[3]={1, 1, 1};
    int indexJ=0;
    int image3D;
    
    /* Constants used */
    int windowratio=3;
    double filterstrength=0.2;
    int kernelratio=3;
    
    int i;
    int block[6]={1, 1, 1, 1, 1, 1};
    int block_size[3];
    double *T;
    double Isize_d[3];
    double dimsJ_d[3];
    double Vsize_d[3];
    double block_size_d[3];
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
    if((ndimsI<2)||(ndimsI>3)) { mexErrMsgTxt("Image must be 2D or 3D"); }
    dimsI= mxGetDimensions(prhs[0]);
    Isize[0]=dimsI[0]; Isize[1]=dimsI[1];
    if(ndimsI==3) { Isize[2]=dimsI[2]; }
    
    if(Isize[2]>3) { image3D=1; } else { image3D=0; }
    
    /* Connect input image */
    I=(double *)mxGetData(prhs[0]);
    
    /* Check input vector dimensions */
    ndimsV=mxGetNumberOfDimensions(prhs[1]);
    if(ndimsV!=2) { mexErrMsgTxt("input vectors must be 2D"); }
    dimsV= mxGetDimensions(prhs[1]);
    Vsize[0]=dimsV[0]; Vsize[1]=dimsV[1];
    
    /* Connect vectors */
    V=(double *)mxGetData(prhs[1]);
    
    /* Set Values */
    T=(double *)mxGetData(prhs[2]);
    kernelratio=(int)T[0];
    
    /* Set Values */
    T=(double *)mxGetData(prhs[3]);
    windowratio=(int)T[0];
    
    /* Set Values */
    T=(double *)mxGetData(prhs[4]);
    filterstrength=T[0];
    
    Nthreadsd=(double *)mxGetData(prhs[5]);
    
    
    /* Calculate block size */
    if(image3D==0) {
        block[0]=windowratio+kernelratio;
        block[1]=windowratio+kernelratio;
        block[2]=dimsI[0]-(windowratio+kernelratio)-1;
        block[3]=dimsI[1]-(windowratio+kernelratio)-1;
        
        block_size[0]=block[2]-block[0]+1;
        block_size[1]=block[3]-block[1]+1;
        
        dimsJ[0]=block_size[0];
        dimsJ[1]=block_size[1];
        dimsJ[2]=Isize[2];
    }
    else {
        block[0]=windowratio+kernelratio;
        block[1]=windowratio+kernelratio;
        block[2]=windowratio+kernelratio;
        
        block[3]=dimsI[0]-(windowratio+kernelratio)-1;
        block[4]=dimsI[1]-(windowratio+kernelratio)-1;
        block[5]=dimsI[2]-(windowratio+kernelratio)-1;
        
        block_size[0]=block[3]-block[0]+1;
        block_size[1]=block[4]-block[1]+1;
        block_size[2]=block[5]-block[2]+1;
        
        dimsJ[0]=block_size[0];
        dimsJ[1]=block_size[1];
        dimsJ[2]=block_size[2];
    }
    
    plhs[0] = mxCreateNumericArray(ndimsJ, dimsJ, mxDOUBLE_CLASS, mxREAL);
    J=(double *)mxGetData(plhs[0]);

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
        dimsJ_d[i]=(double)dimsJ[i];
        Vsize_d[i]=(double)Vsize[i];
        block_size_d[i]=(double)block_size[i];
        block_d[i] =(double)block[i];
        block_d[i+3] =(double)block[i+3];
    }
    par_d[0] =(double)kernelratio;
    par_d[1] =(double)windowratio;
    par_d[2] =(double)filterstrength;
    par_d[3] =(double)image3D;

    for (i=0; i<Nthreads; i++) {
        /*  Make Thread ID  */
        ThreadID1= (double *)malloc( 1* sizeof(double) );
        ThreadID1[0]=i;
        ThreadID[i]=ThreadID1;
        
        /*  Make Thread Structure  */
        ThreadArgs1 = (double **)malloc( 11* sizeof( double * ) );
        ThreadArgs1[0]=Isize_d;
        ThreadArgs1[1]=dimsJ_d;
        ThreadArgs1[2]=Vsize_d;
        ThreadArgs1[3]=block_size_d;
        ThreadArgs1[4]=par_d;
        ThreadArgs1[5]=block_d;
        ThreadArgs1[6]=I;
        ThreadArgs1[7]=J;
        ThreadArgs1[8]=V;
        ThreadArgs1[9]=ThreadID[i];
        ThreadArgs1[10]=Nthreadsd;
        /* Start a Thread  */
        ThreadArgs[i]=ThreadArgs1;
        
        
        #ifdef _WIN32
            ThreadList[i] = (HANDLE)_beginthreadex( NULL, 0, &filter_multi_threaded, ThreadArgs[i] , 0, NULL );
        #else
            pthread_create((pthread_t*)&ThreadList[i], NULL, (void *) &filter_multi_threaded, ThreadArgs[i]);
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
}

