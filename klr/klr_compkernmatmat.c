/* -------------------------------------------------------------------
 * KLR_COMPKERNMATMAT
 *
 * Matrix-matrix product Y = K*X, K block-diagonal kernel matrix.
 * X is size N*P-by-Q, K has P N-by-N blocks. The blocks are K(j) =
 * V(p)*M(l(p)) + S(p), where V(p), S(p) are positive from VVEC, SVEC.
 * S(p) can be 0. M(l)
 * are matrices stored in MBUFF of size N-by-(N*?). M(1), M(2) are
 * stored in lower, upper triangle of leftmost block, M(3), M(4)
 * in the second block, etc. The diagonals of the M(l) are columns
 * of MDIAG. They overwrite the corr. diag. in MBUFF on demand.
 * p -> l coded in L2P, containing chunks NUM,p_1,...,p_NUM for
 * subseq. values l. Values for p start with 1.
 *
 * If SZ is given, we compute Y = K(1:SZ,1:SZ)*X instead, where 1:SZ
 * is index into dataset (size N). In this case, X and Y have size
 * SZ*P-by-Q. If SZ not given, then SZ==N.
 *
 * We need a temp. buffer of size 2*SZ*Q*MAXNUM, where MAXNUM is the
 * max. value of the NUM fields in L2P. This buffer can be provided
 * in TMPBUFF, otherwise it is alloc. locally.
 *
 * Input:
 * - N:       Dataset size
 * - X:       Input matrix
 * - VVEC:    S.a.
 * - SVEC:    S.a.
 * - MBUFF:   S.a.
 * - MDIAG:   S.a.
 * - L2P:     S.a.
 * - SZ:      S.a. Optional
 * - TMPBUFF: S.a. Optional
 *
 * Return:
 * - Y:     Return matrix
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

/* Main function KLR_COMPKERNMATVECT */

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int i,j,n,p,l,nl,num,maxnum,pos,lenl2p,pind,ione=1,mpos,q,tempi,sz;
  double zero=0.0,one=1.0,vval,sval;
  double* mbuff,*tempP,*ymat,*xP;
  const double* xmat,*vvec,*mdiag,*l2p,*svec;
  char uplo[2],side[2];
  bool ownTmp;

  /* Read arguments */
  if (nrhs<7)
    mexErrMsgTxt("Not enough input arguments");
  if (nlhs!=1)
    mexErrMsgTxt("Returns one argument");
  if ((n=getScalInt(prhs[0],"N"))<=0)
    mexErrMsgTxt("Wrong argument N");
  sz=n;
  if (nrhs>7) {
    sz=getScalInt(prhs[7],"SZ");
    if (sz<1 || sz>n) mexErrMsgTxt("Wrong argument SZ");
  }
  if (!mxIsDouble(prhs[1]) || (p=mxGetM(prhs[1]))%sz!=0 ||
      (q=mxGetN(prhs[1]))==0)
    mexErrMsgTxt("X has wrong size");
  p/=sz;
  xmat=mxGetPr(prhs[1]);
  if (getVecLen(prhs[2],"VVEC")!=p)
    mexErrMsgTxt("VVEC has wrong size");
  vvec=mxGetPr(prhs[2]);
  if (getVecLen(prhs[3],"SVEC")!=p)
    mexErrMsgTxt("SVEC has wrong size");
  svec=mxGetPr(prhs[3]);
  if (!mxIsDouble(prhs[4]) || mxGetM(prhs[4])!=n ||
      (i=mxGetN(prhs[4]))%n!=0)
    mexErrMsgTxt("MBUFF has wrong size");
  mbuff=mxGetPr(prhs[4]);
  i/=n; i*=2;
  if (!mxIsDouble(prhs[5]) || mxGetM(prhs[5])!=n ||
      (nl=mxGetN(prhs[5]))>i)
    mexErrMsgTxt("MDIAG has wrong size");
  mdiag=mxGetPr(prhs[5]);
  if ((lenl2p=getVecLen(prhs[6],"L2P"))==0)
    mexErrMsgTxt("Wrong argument L2P");
  l2p=mxGetPr(prhs[6]);
  tempP=0; ownTmp=true;
  if (nrhs>8) {
    if (!mxIsDouble(prhs[8])) mexErrMsgTxt("Wrong argument TMPBUFF");
    tempP=mxGetPr(prhs[8]); ownTmp=false;
  }

  /* Allocate memory */
  plhs[0]=mxCreateDoubleMatrix(sz*p,q,mxREAL); /* Y */
  ymat=mxGetPr(plhs[0]);
  maxnum=0;
  for (pos=0; pos<lenl2p; ) {
    num=(int) l2p[pos++]; pos+=num;
    if (num>maxnum) maxnum=num;
  }
  if (!ownTmp && mxGetM(prhs[8])*mxGetN(prhs[8])<sz*q*maxnum*2)
    ownTmp=true; /* Buffer too small: Allocate */
  if (ownTmp) {
    tempP=(double*) mxMalloc(sz*q*maxnum*2*sizeof(double));
    mexPrintf("klr_compkernmatmat: need to alloc. tmpbuff, sz=%d\n",sz*q*maxnum*2); /* DEBUG! */
  }

  /* Main loop */
  side[0]='L'; side[1]=0; uplo[0]='L'; uplo[1]=0;
  for (pos=l=mpos=0; l<nl; l++) {
    num=(int) l2p[pos++];
    /* Copy */
    for (i=0; i<num; i++) {
      pind=((int) l2p[pos+i])-1;
      for (j=0; j<q; j++)
	memmove(tempP+(sz*(i*q+j)),xmat+(sz*(pind+j*p)),sz*sizeof(double));
    }
    /* Correct diagonal of M(l) */
    tempi=n+1; /* stride for diag. */
    BLASFUNC(dcopy) (&sz,mdiag+(l*n),&ione,mbuff+mpos,&tempi);
    /* Multiply with kernel matrix */
    maxnum=num*q; /* Cols of temp. matrix */
    BLASFUNC(dsymm) (side,uplo,&sz,&maxnum,&one,mbuff+mpos,&n,tempP,&sz,&zero,
		     tempP+(maxnum*sz),&sz);
    /* Copy back, premultiply */
    tempi=sz*q;
    for (i=0; i<num; i++) {
      pind=((int) l2p[pos+i])-1;
      vval=vvec[pind]; sval=svec[pind];
      xP=tempP+(tempi*(i+num));
      for (j=0; j<tempi; j++)
	*(xP++)=vval*(*xP)+sval;
      for (j=0; j<q; j++)
	memmove(ymat+(sz*(pind+j*p)),tempP+(sz*((i+num)*q+j)),
		sz*sizeof(double));
    }
    /* Flip UPLO */
    if (uplo[0]=='L') uplo[0]='U';
    else {
      uplo[0]='L'; mpos+=(n*n); /* next block */
    }
    pos+=num;
  }

  if (ownTmp) mxFree((void*) tempP);
}
