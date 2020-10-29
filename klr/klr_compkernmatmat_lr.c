/* -------------------------------------------------------------------
 * KLR_COMPKERNMATMAT_LR
 *
 * Matrix-matrix product Y = K*X, K block-diagonal kernel matrix.
 * Does the same as KLR_COMPKERNMATMAT, but the matrix blocks
 * M^(l) are replaced by a low rank approximation, whose diagonal
 * is corrected to the original one. See low rank documentation or
 * paper for details, and KLR_COMPKERNMATVECT_LR.
 *
 * NOTE: We always compute full MVMs with vectors of size N and use
 * SZ only at start/end to extract rel. parts. This is slow if SZ is
 * much smaller than N.
 *
 * NOTE: Could do with less memory for TMPBUFF!
 *
 * Input:
 * - N:       Dataset size
 * - X:       Input matrix
 * - VVEC:    S.a.
 * - SVEC:    S.a.
 * - MBUFF:   Contains the N-by-(d_l) factors L^(l) from left to
 *            right
 * - MDIAG:   True diagonals of M^(l) as cols
 * - LLDIAG:  Diagonals of P^(l) L^(l) [P^(l) L^(l)]^T as cols
 * - PERM:    N-by-?, l-th col. is P^(l) permutation
 * - ACTSZ:   Vector cont. active set sizes d_l
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

/* Main function KLR_COMPKERNMATMAT_LR */

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int i,j,k,n,p,l,nl,num,maxnum,pos,lenl2p,ione=1,mpos,q,sz,szi,tempi;
  double zero=0.0,one=1.0,vval,sval;
  double* mbuff,*tempP,*ymat,*yP;
  const double* xmat,*vvec,*mdiag,*l2p,*lldiag,*perm,*actsz,*xP,*mdP,
    *ldP,*svec;
  char trans[2],trans2[2];
  bool ownTmp;
  int* iperm,*pind;

  /* Read arguments */
  if (nrhs<10)
    mexErrMsgTxt("Not enough input arguments");
  if (nlhs!=1)
    mexErrMsgTxt("Returns one argument");
  if ((n=getScalInt(prhs[0],"N"))<=0)
    mexErrMsgTxt("Wrong argument N");
  sz=n;
  if (nrhs>10) {
    sz=getScalInt(prhs[10],"SZ");
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
  if (!mxIsDouble(prhs[6]) || mxGetM(prhs[6])!=n ||
      (mxGetN(prhs[6])!=nl))
    mexErrMsgTxt("LLDIAG has wrong size");
  lldiag=mxGetPr(prhs[6]);
  if (!mxIsDouble(prhs[7]) || mxGetM(prhs[7])!=n ||
      (mxGetN(prhs[7])!=nl))
    mexErrMsgTxt("PERM has wrong size");
  perm=mxGetPr(prhs[7]);
  if (getVecLen(prhs[8],"ACTSZ")!=nl)
    mexErrMsgTxt("ACTSZ has wrong size");
  actsz=mxGetPr(prhs[8]);
  if ((lenl2p=getVecLen(prhs[9],"L2P"))==0)
    mexErrMsgTxt("Wrong argument L2P");
  l2p=mxGetPr(prhs[9]);
  tempP=0; ownTmp=true;
  if (nrhs>11) {
    if (!mxIsDouble(prhs[11]))
      mexErrMsgTxt("Wrong argument TMPBUFF");
    tempP=mxGetPr(prhs[11]); ownTmp=false;
  }

  /* Allocate memory */
  plhs[0]=mxCreateDoubleMatrix(sz*p,q,mxREAL); /* Y */
  ymat=mxGetPr(plhs[0]);
  maxnum=0;
  for (pos=0; pos<lenl2p; ) {
    num=(int) l2p[pos++]; pos+=num;
    if (num>maxnum) maxnum=num;
  }
  if (!ownTmp && mxGetM(prhs[11])*mxGetN(prhs[11])<2*n*maxnum*q)
    ownTmp=true; /* Buffer too small: Allocate another one */
  if (ownTmp)
    tempP=(double*) mxMalloc(2*n*q*maxnum*sizeof(double));
  iperm=(int*) mxMalloc(n*sizeof(int));
  pind=(int*) mxMalloc(maxnum*sizeof(int));

  /* Main loop */
  trans[1]=0; trans2[0]='N'; trans2[1]=0;
  for (pos=l=mpos=0; l<nl; l++) {
    num=(int) l2p[pos++];
    for (i=0; i<num; i++) pind[i]=((int) l2p[pos++])-1; /* p_k for l */
    for (i=0; i<n; i++) iperm[i]=((int) perm[l*n+i])-1; /* P^(l) */
    szi=(int) actsz[l]; /* Size of I_l */
    /* Copy and permute by P^T
       For y = P^T x: y[i] = x[iperm[i]] */
    fillVec(tempP,2*n*q*num,0.0);
    tempi=sz*p;
    for (i=0; i<n; i++)
      if ((j=iperm[i])<sz)
	for (k=0; k<num; k++)
	  BLASFUNC(dcopy) (&q,xmat+(pind[k]*sz+j),&tempi,tempP+(n*q*k+i),&n);
    /* Matrix-matrix multiplications.
       Use 2nd block of 'tempP' to store intermediate (would need less
       space) */
    tempi=q*num;
    trans[0]='T';
    BLASFUNC(dgemm) (trans,trans2,&szi,&tempi,&n,&one,mbuff+mpos,&n,tempP,&n,
		     &zero,tempP+(n*q*num),&szi);
    trans[0]='N';
    BLASFUNC(dgemm) (trans,trans2,&n,&tempi,&szi,&one,mbuff+mpos,&n,
		     tempP+(n*q*num),&szi,&zero,tempP,&n);
    /* Permute by P and copy back to 'yvec'.
       For y = P x: y[iperm[i]] = x[i] */
    tempi=sz*p;
    for (i=0; i<n; i++)
      if ((j=iperm[i])<sz)
	for (k=0; k<num; k++)
	  BLASFUNC(dcopy) (&q,tempP+(n*q*k+i),&n,ymat+(pind[k]*sz+j),&tempi);
    /* Diagonal correction (no permutation), premult. with v_p */
    for (k=0; k<num; k++)
      for (j=0; j<q; j++) {
	tempi=sz*(pind[k]+p*j);
	yP=ymat+tempi; xP=xmat+tempi;
	mdP=mdiag+(l*n); ldP=lldiag+(l*n);
	vval=vvec[pind[k]]; sval=svec[pind[k]];
	for (i=0; i<sz; i++)
	  *(yP++)=vval*((*yP)+(*(xP++))*((*(mdP++))-(*(ldP++))))+sval;
      }
    mpos+=(n*szi); /* Next L^(l) */
  }

  if (ownTmp) mxFree((void*) tempP);
  mxFree((void*) iperm); mxFree((void*) pind);
}
