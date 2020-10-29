/* -------------------------------------------------------------------
 * KLR_COMPMATVECT
 *
 * Matrix-vector product Y = A*X, A symmetric, stored in lower
 * triangle (LOWER==1) or upper triangle (LOWER==0) of A. If INDY,
 * INDX are given, the matrix is A(INDY,INDX) instead of A. If only
 * INDY is given, then INDX==INDY.
 * Alternatively, a vector [0; NS] can be passed for INDY (INDX must
 * not be used then), meaning that 1:NS is taken for INDY.
 *
 * Input:
 * - A:     Symmetric matrix
 * - X:     Input vector
 * - LOWER: S.a.
 * - INDY:  S.a. Optional
 * - INDX:  S.a. Optional
 *
 * Return:
 * - Y:     Return vector
 * -------------------------------------------------------------------
 * Matlab MEX Function
 * Author: Matthias Seeger
 * ------------------------------------------------------------------- */

#define MATLAB_VER65

#include <math.h>
#include "mex.h"
#include "mex_helper.h"
#include "blas_headers.h"
#include "klr_compmatvectind.h"

char errMsg[200];

/* Main function KLR_COMPMATVECT */

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int i,j,n,nx,step1,step2,posi,posj,ld;
  double temp,zero=0.0,one=1.0,sum;
  double* yP;
  int* indxi,*indyi;
  const double* xvec,*indy=0,*indx=0,*xP,*mat;
  char uplo[2];

  /* Read arguments */
  if (nrhs<3)
    mexErrMsgTxt("Not enough input arguments");
  if (nlhs!=1)
    mexErrMsgTxt("Returns one argument");
  if (!mxIsDouble(prhs[0]) || (n=mxGetM(prhs[0]))!=mxGetN(prhs[0]))
    mexErrMsgTxt("Wrong argument A");
  mat=mxGetPr(prhs[0]);
  ld=n;
  xvec=mxGetPr(prhs[1]);
  i=getScalInt(prhs[2],"LOWER");
  uplo[1]=0;
  if (i!=0) uplo[0]='L';
  else uplo[0]='U';
  nx=n;
  if (nrhs>=4) {
    i=getVecLen(prhs[3],"INDY");
    if (i==0) mexErrMsgTxt("INDY invalid");
    indy=mxGetPr(prhs[3]);
    if (indy[0]==0.0) {
      /* Special case */
      if (i!=2) mexErrMsgTxt("INDY invalid");
      if ((n=(int) indy[1])<=0) mexErrMsgTxt("INDY invalid");
      nx=n;
      indy=0; /* not needed */
    } else {
      if (i>ld) mexErrMsgTxt("INDY invalid");
      n=nx=i;
      if (nrhs>=5) {
	i=getVecLen(prhs[4],"INDX");
	if (i==0 || i>ld) mexErrMsgTxt("INDX invalid");
	indx=mxGetPr(prhs[4]);
	nx=i;
      }
    }
  }
  if (getVecLen(prhs[1],"X")!=nx)
    mexErrMsgTxt("X has wrong size");

  plhs[0]=mxCreateDoubleMatrix(n,1,mxREAL); /* Y */
  if (indy==0) {
    /* Call DSYMV */
    i=1;
    BLASFUNC(dsymv) (uplo,&n,&one,mat,&ld,xvec,&i,&zero,mxGetPr(plhs[0]),&i);
  } else {
    /* MV with INDY, INDX indexes */
    klr_compmatvectind(xvec,mxGetPr(plhs[0]),mat,ld,(uplo[0]=='L'),indy,n,
		       indx,nx);
  }
}
