/* -------------------------------------------------------------------
 * KLR_ADDSIGSQ
 *
 * Y = Y + SIGSQ*P_data*X
 *
 * Input:
 * - Y:       Input/output vector.
 * - X:       Input vector
 * - SIGSQ:   Scalar
 * - NN:      Dataset size
 * - NC:      Number classes
 * -------------------------------------------------------------------
 * Matlab MEX Function
 * Author: Matthias Seeger
 * ------------------------------------------------------------------- */

#define MATLAB_VER65

#include <math.h>
#include "mex.h"
#include "mex_helper.h"

char errMsg[200];

/* Main function KLR_MEXMULV */

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int i,j,nn,nc,n,ione=1;
  double sigsq,sum;
  const double* xvec,*xP;
  double* yvec,*yP;

  /* Read arguments */
  if (nrhs<5)
    mexErrMsgTxt("Not enough input arguments");
  if ((nn=getScalInt(prhs[3],"NN"))<=0)
    mexErrMsgTxt("NN wrong");
  if ((nc=getScalInt(prhs[4],"NC"))<=0)
    mexErrMsgTxt("NC wrong");
  n=nn*nc;
  if (getVecLen(prhs[0],"Y")!=n)
    mexErrMsgTxt("Y has wrong size");
  yvec=mxGetPr(prhs[0]);
  if (getVecLen(prhs[1],"X")!=n)
    mexErrMsgTxt("X has wrong size");
  xvec=mxGetPr(prhs[1]);
  sigsq=getScalar(prhs[2],"SIGSQ");

  xP=xvec; yP=yvec;
  for (j=0; j<nc; j++) {
    for (i=0,sum=0.0; i<nn; i++) sum+=*(xP++);
    sum*=sigsq;
    for (i=0; i<nn; i++) (*(yP++))+=sum;
  }
}
