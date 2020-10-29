/* -------------------------------------------------------------------
 * KLR_MEXMULV
 *
 * Does the same as KLR_MULV, but in-place: X is overwritten by
 * solution Y.
 *
 * Input:
 * - X:       Input/output vector, overwritten by Y
 * - TRS:     See KLR_MULV 
 * - NN:      Dataset size
 * - NC:      Number classes
 * - SQPI:    Value of KLR_INTERN.SQPI
 * - PI:      Value of KLR_INTERN.PI
 * -------------------------------------------------------------------
 * Matlab MEX Function
 * Author: Matthias Seeger
 * ------------------------------------------------------------------- */

#define MATLAB_VER65

#include <math.h>
#include "mex.h"
#include "mex_helper.h"
#include "blas_headers.h"

char errMsg[200];

/* Main function KLR_MEXMULV */

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int i,j,nn,nc,n,ione=1;
  double one=1.0;
  const double* pi,*sqpi,*piP;
  double* xvec,*xP,*tvec,*tP;
  bool trs;

  /* Read arguments */
  if (nrhs<6)
    mexErrMsgTxt("Not enough input arguments");
  if ((nn=getScalInt(prhs[2],"NN"))<=0)
    mexErrMsgTxt("NN wrong");
  if ((nc=getScalInt(prhs[3],"NC"))<=0)
    mexErrMsgTxt("NC wrong");
  n=nn*nc;
  if (getVecLen(prhs[0],"X")!=n)
    mexErrMsgTxt("X has wrong size");
  xvec=mxGetPr(prhs[0]);
  trs=(getScalInt(prhs[1],"TRS")!=0);
  if (getVecLen(prhs[4],"SQPI")!=n ||
      getVecLen(prhs[5],"PI")!=n)
    mexErrMsgTxt("PI / SQPI wrong size");
  pi=mxGetPr(prhs[5]); sqpi=mxGetPr(prhs[4]);

  tvec=(double*) mxMalloc(nn*sizeof(double));
  fillVec(tvec,nn,0.0);
  if (!trs) {
    /* Y = V*X */
    xP=xvec; piP=sqpi;
    for (i=0; i<n; i++)
      (*(xP++))*=(*(piP++));
    xP=xvec;
    for (j=0; j<nc; j++,xP+=nn)
      BLASFUNC(daxpy) (&nn,&one,xP,&ione,tvec,&ione);
    xP=xvec; piP=pi;
    for (j=0; j<nc; j++) {
      tP=tvec;
      for (i=0; i<nn; i++)
	(*(xP++))-=(*(piP++))*(*(tP++));
    }
  } else {
    /* Y = V'*X */
    xP=xvec; piP=pi;
    for (j=0; j<nc; j++) {
      tP=tvec;
      for (i=0; i<nn; i++)
	(*(tP++))+=(*(piP++))*(*(xP++));
    }
    xP=xvec; piP=sqpi;
    for (j=0; j<nc; j++) {
      tP=tvec;
      for (i=0; i<nn; i++)
	(*(xP++))=(*(piP++))*((*xP)-(*(tP++)));
    }
  }
  mxFree((void*) tvec);
}
