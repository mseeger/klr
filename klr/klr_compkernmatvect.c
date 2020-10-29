/* -------------------------------------------------------------------
 * KLR_COMPKERNMATVECT
 *
 * Matrix-vector product Y = K*X, K block-diagonal kernel matrix.
 * X is size N*P, K has P N-by-N blocks. The blocks are K(j) =
 * V(p)*M(l(p)) + S(p), where V(p), S(p) are positive from VVEC,
 * SVEC. S(p) can be 0. M(l) are matrices stored in MBUFF of size
 * N-by-(N*?). M(1), M(2) are
 * stored in lower, upper triangle of leftmost block, M(3), M(4)
 * in the second block, etc. The diagonals of the M(l) are columns
 * of MDIAG. They overwrite the corr. diag. in MBUFF on demand.
 * p -> l coded in L2P, containing chunks num,p_1,...,p_num for
 * subseq. values l. Values for p start with 1. Requires that
 * p_1 < p_2 < ... < p_num.
 *
 * INDY, INDX have the same meaning as in KLR_COMPMATVECT. If INDY
 * has the form [0; NY], then INDX is ignored (even if given). If
 * INDY==[], it is set to the default [0; N], and INDX is ignored.
 *
 * We need a temp. buffer of size 2*N*MAXNUM, where MAXNUM is the
 * max. value of the 'num' fields in L2P. This buffer can be provided
 * in TMPBUFF, otherwise it is alloc. locally.
 * If INDY, INDX are used of size NY, NX, replace 2*N by NX+NY.
 *
 * Input:
 * - N:       Dataset size
 * - X:       Input vector
 * - VVEC:    S.a.
 * - SVEC:    S.a.
 * - MBUFF:   S.a.
 * - MDIAG:   S.a.
 * - L2P:     S.a.
 * - INDY:    S.a. Optional
 * - INDX:    S.a. Optional
 * - TMPBUFF: S.a. Optional
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

/* Main function KLR_COMPKERNMATVECT */

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int i,j,n,p,l,nl,num,maxnum,pos,lenl2p,pind,ione=1,mpos,ny,nx;
  double zero=0.0,one=1.0,vval,sval;
  double* mbuff,*tempP,*yvec,*yP;
  const double* xvec,*vvec,*mdiag,*l2p,*indy=0,*indx=0,*svec,*xP;
  char uplo[2],side[2];
  bool ownTmp;

  /* Read arguments */
  if (nrhs<7)
    mexErrMsgTxt("Not enough input arguments");
  if (nlhs!=1)
    mexErrMsgTxt("Returns one argument");
  if ((n=getScalInt(prhs[0],"N"))<=0)
    mexErrMsgTxt("Wrong argument N");
  ny=nx=n;
  if (nrhs>7) {
    if ((i=getVecLen(prhs[7],"INDY"))>0) {
      indy=mxGetPr(prhs[7]);
      if (indy[0]==0.0) {
	/* Special case [0; NY] */
	if (i!=2) mexErrMsgTxt("INDY invalid");
	ny=(int) indy[1];
	if (ny<1 || ny>n) mexErrMsgTxt("INDY invalid");
	nx=ny;
	indy=0; /* not needed */
      } else {
	if (i>n) mexErrMsgTxt("INDY invalid");
	ny=nx=i;
	if (nrhs>8) {
	  if ((i=getVecLen(prhs[8],"INDX"))>0) {
	    if (i>n) mexErrMsgTxt("INDX invalid");
	    indx=mxGetPr(prhs[8]);
	    nx=i;
	  }
	}
      }
    }
  }
  if ((p=getVecLen(prhs[1],"X"))%nx!=0)
    mexErrMsgTxt("X has wrong size");
  p/=nx;
  xvec=mxGetPr(prhs[1]);
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
  if (nrhs>9) {
    if (!mxIsDouble(prhs[9]))
      mexErrMsgTxt("Wrong argument TMPBUFF");
    tempP=mxGetPr(prhs[9]); ownTmp=false;
  }

  /* Allocate memory */
  plhs[0]=mxCreateDoubleMatrix(ny*p,1,mxREAL); /* Y */
  yvec=mxGetPr(plhs[0]);
  maxnum=0;
  for (pos=0; pos<lenl2p; ) {
    num=(int) l2p[pos++]; pos+=num;
    if (num>maxnum) maxnum=num;
  }
  if (!ownTmp && mxGetM(prhs[9])*mxGetN(prhs[9])<(ny+nx)*maxnum)
    ownTmp=true; /* Buffer too small: Allocate another one */
  if (ownTmp) {
    tempP=(double*) mxMalloc((ny+nx)*maxnum*sizeof(double));
    mexPrintf("klr_compkernmatvec: need to alloc. tmpbuff\n"); /* DEBUG! */
  }

  /* Main loop */
  side[0]='L'; side[1]=0; uplo[0]='L'; uplo[1]=0;
  for (pos=l=mpos=0; l<nl; l++) {
    num=(int) l2p[pos++];
    /* Copy */
    for (i=0; i<num; i++) {
      pind=((int) l2p[pos+i])-1;
      memmove(tempP+(i*nx),xvec+(pind*nx),nx*sizeof(double));
    }
    /* Correct diagonal of M(l) */
    maxnum=n+1; /* stride for diag. */
    BLASFUNC(dcopy) (&n,mdiag+(l*n),&ione,mbuff+mpos,&maxnum);
    /* Multiply with kernel matrix */
    if (indy==0) {
      /* NX==NY here */
      BLASFUNC(dsymm) (side,uplo,&ny,&num,&one,mbuff+mpos,&n,tempP,&nx,&zero,
		       tempP+(num*nx),&ny);
    } else {
      klr_compmatmatind(tempP,tempP+(num*nx),num,mbuff+mpos,n,(uplo[0]=='L'),
			indy,ny,indx,nx);
    }
    /* Copy back, premultiply, add s_p */
    for (i=0; i<num; i++) {
      pind=((int) l2p[pos+i])-1;
      yP=yvec+(pind*ny); xP=tempP+(num*nx+i*ny);
      vval=vvec[pind]; sval=svec[pind];
      for (j=0; j<ny; j++)
	*(yP++)=vval*(*(xP++))+sval;
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
