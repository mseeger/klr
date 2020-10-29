/* -------------------------------------------------------------------
 * KLR_COMPMATPHIT
 *
 * Matrix-matrix product Y = X*PHI', where PHI is the indicator matrix
 * for a class hierarchy. X is of size N-by-P. The hierarchy is
 * described in CN (size P+1). Nodes are labelled 0,...,P, 0 is the
 * root which is not repres. in X. CN(j) is the number of children of
 * j, which are CS(j)+1,...,CS(j)+CN(j). CS is not required, because
 * the labelling is contiguous. CN(j)==0 means that j is a leaf node.
 * The input matrix X is overwritten by the result matrix Y.
 *
 * Input:
 * - X:  Input matrix. Overwritten by result Y
 * - CN: S.a.
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

/* Main function KLR_COMPMATPHIT */

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int i,j,n,p,cn,ione=1;
  double one=1.0,temp;
  double* xmat,*ymP;
  const double* cnvec,*yvP;

  /* Read arguments */
  if (nrhs<2)
    mexErrMsgTxt("Not enough input arguments");
  if (!mxIsDouble(prhs[0]) || (n=mxGetM(prhs[0]))==0 ||
      (p=mxGetN(prhs[0]))==0)
    mexErrMsgTxt("Wrong argument X");
  xmat=mxGetPr(prhs[0]);
  if (getVecLen(prhs[1],"CN")!=p+1)
    mexErrMsgTxt("CN has wrong size");
  cnvec=mxGetPr(prhs[1]);

  /* Main loop */
  cn=(int) *(cnvec++);
  ymP=xmat+(n*cn);
  if (n>1) {
    for (i=0; i<p; i++)
      if ((cn=(int) *(cnvec++))>0) {
	yvP=xmat+(n*i);
	for (j=0; j<cn; j++) {
	  BLASFUNC(daxpy) (&n,&one,yvP,&ione,ymP,&ione);
	  ymP+=n;
	}
      }
  } else {
    for (i=0; i<p; i++)
      if ((cn=(int) *(cnvec++))>0) {
	temp=xmat[i];
	for (j=0; j<cn; j++)
	  (*(ymP++))+=temp;
      }
  }
}
