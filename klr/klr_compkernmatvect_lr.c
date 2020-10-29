/* -------------------------------------------------------------------
 * KLR_COMPKERNMATVECT_LR
 *
 * Matrix-vector product Y = K*X, K block-diagonal kernel matrix.
 * Does the same as KLR_COMPKERNMATVECT, but the matrix blocks
 * M^(l) are replaced by a low rank approximation, whose diagonal
 * is corrected to the original one. See low rank documentation or
 * paper for details.
 *
 * NOTE: We always compute full MVMs with vectors of size N and use
 * INDY, INDX only at start/end to extract rel. parts. This is
 * slow if their sizes are much smaller than N.
 *
 * NOTE: Could do with less memory for TMPBUFF!
 *
 * Input:
 * - N:       Dataset size
 * - X:       Input vector
 * - VVEC:    S.a.
 * - SVEC:    S.a.
 * - MBUFF:   Contains the N-by-(d_l) factors L^(l) from left to
 *            right
 * - MDIAG:   True diagonals of M^(l) as cols
 * - LLDIAG:  Diagonals of P^(l) L^(l) [P^(l) L^(l)]^T as cols
 * - PERM:    N-by-?, l-th col. is P^(l) permutation
 * - ACTSZ:   Vector cont. active set sizes d_l
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

char errMsg[200];

/* Main function KLR_COMPKERNMATVECT_LR */

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int i,j,k,n,p,l,nl,num,maxnum,pos,lenl2p,ione=1,mpos,ny,nx,szi;
  double zero=0.0,one=1.0,vval,sval;
  double* mbuff,*tempP,*yvec,*yP;
  const double* xvec,*vvec,*mdiag,*l2p,*indy=0,*indx=0,*lldiag,
    *perm,*actsz,*xP,*mdP,*ldP,*svec;
  char trans[2],trans2[2];
  bool ownTmp;
  int* iperm,*pind,*iindy=0,*iindx=0;
  bool* tickoff=0;

  /* Read arguments */
  if (nrhs<10)
    mexErrMsgTxt("Not enough input arguments");
  if (nlhs!=1)
    mexErrMsgTxt("Returns one argument");
  if ((n=getScalInt(prhs[0],"N"))<=0)
    mexErrMsgTxt("Wrong argument N");
  ny=nx=n;
  if (nrhs>10) {
    if ((i=getVecLen(prhs[10],"INDY"))>0) {
      indy=mxGetPr(prhs[10]);
      if (indy[0]==0.0) {
	/* Special case [0; NY] */
	if (i!=2) mexErrMsgTxt("INDY invalid");
	ny=(int) indy[1];
	if (ny<1 || ny>n) mexErrMsgTxt("INDY invalid");
	nx=ny;
	indy=0; /* not needed */
      } else {
	if (i>n) mexErrMsgTxt("INDY invalid");
	n=nx=i;
	if (nrhs>11) {
	  if ((i=getVecLen(prhs[11],"INDX"))>0) {
	    if (i>n) mexErrMsgTxt("INDX invalid");
	    indx=mxGetPr(prhs[11]);
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
  if (nrhs>12) {
    if (!mxIsDouble(prhs[12]))
      mexErrMsgTxt("Wrong argument TMPBUFF");
    tempP=mxGetPr(prhs[12]); ownTmp=false;
  }

  /* Allocate memory */
  plhs[0]=mxCreateDoubleMatrix(ny*p,1,mxREAL); /* Y */
  yvec=mxGetPr(plhs[0]);
  maxnum=0;
  for (pos=0; pos<lenl2p; ) {
    num=(int) l2p[pos++]; pos+=num;
    if (num>maxnum) maxnum=num;
  }
  if (!ownTmp && mxGetM(prhs[12])*mxGetN(prhs[12])<2*n*maxnum)
    ownTmp=true; /* Buffer too small: Allocate another one */
  if (ownTmp)
    tempP=(double*) mxMalloc(2*n*maxnum*sizeof(double));
  iperm=(int*) mxMalloc(n*sizeof(int));
  pind=(int*) mxMalloc(maxnum*sizeof(int));
  if (indy!=0) {
    tickoff=(bool*) mxMalloc(n*sizeof(bool));
    iindy=(int*) mxMalloc(ny*sizeof(int));
    for (i=0; i<ny; i++) iindy[i]=((int) indy[i])-1;
    if (indx==0)
      iindx=iindy;
    else {
      iindx=(int*) mxMalloc(nx*sizeof(int));
      for (i=0; i<nx; i++) iindx[i]=((int) indx[i])-1;
    }
  }

  /* Main loop */
  trans[1]=0; trans2[0]='N'; trans2[1]=0;
  for (pos=l=mpos=0; l<nl; l++) {
    num=(int) l2p[pos++];
    for (i=0; i<num; i++) pind[i]=((int) l2p[pos++])-1; /* p_k for l */
    for (i=0; i<n; i++) iperm[i]=((int) perm[l*n+i])-1; /* P^(l) */
    szi=(int) actsz[l]; /* Size of I_l */
    /* Copy and permute by P^T
       For y = P^T x: y[i] = x[iperm[i]] */
    fillVec(tempP,2*n*num,0.0);
    if (indy==0) {
      for (i=0; i<n; i++)
	if ((j=iperm[i])<nx)
	  for (k=0; k<num; k++)
	    tempP[k*n+i]=xvec[pind[k]*nx+j];
    } else {
      /* Use intermediate u = I_{.,indx} x: u[indx[i]] = x[i].
	 So: y[j] = x[i] if iperm[j] = indx[i].
	 Intermediate stuff stored in 2nd block of 'tempP' */
      for (i=0; i<n; i++) tickoff[i]=false;
      for (i=0; i<nx; i++) {
	j=iindx[i]; tickoff[j]=true;
	for (k=0; k<num; k++)
	  tempP[(k+num)*n+j]=xvec[pind[k]*nx+i];
      }
      for (i=0; i<n; i++)
	if (tickoff[j=iperm[i]])
	  BLASFUNC(dcopy) (&num,tempP+(num*n+j),&n,tempP+i,&n); /* Move row */
    }
    /* Matrix-matrix multiplications.
       Use 2nd block of 'tempP' to store intermediate (would need less
       space) */
    trans[0]='T';
    BLASFUNC(dgemm) (trans,trans2,&szi,&num,&n,&one,mbuff+mpos,&n,tempP,&n,
		     &zero,tempP+(n*num),&szi);
    trans[0]='N';
    BLASFUNC(dgemm) (trans,trans2,&n,&num,&szi,&one,mbuff+mpos,&n,
		     tempP+(n*num),&szi,&zero,tempP,&n);
    /* Permute by P and copy back to 'yvec'.
       For y = P x: y[iperm[i]] = x[i] */
    if (indy==0) {
      for (i=0; i<n; i++)
	if ((j=iperm[i])<ny)
	  for (k=0; k<num; k++)
	    yvec[pind[k]*ny+j]=tempP[k*n+i];
    } else {
      /* Use intermediate u = P x: u[iperm[i]] = x[i].
	 So: y[j] = x[i] if iperm[i] = indy[j]
	 Intermediate stuff stored in 2nd block of 'tempP' */
      for (i=0; i<n; i++) tickoff[i]=false;
      for (i=0; i<ny; i++)
	tickoff[iindy[i]]=true;
      for (i=0; i<n; i++)
	if (tickoff[j=iperm[i]])
	  BLASFUNC(dcopy) (&num,tempP+i,&n,tempP+(num*n+j),&n);
      for (i=0; i<ny; i++) {
	j=iindy[i]-1;
	for (k=0; k<num; k++)
	  yvec[pind[k]*ny+i]=tempP[(k+num)*n+j];
      }
    }
    /* Diagonal correction (no permutation), premult. with v_p */
    if (indy==0) {
      for (k=0; k<num; k++) {
	j=pind[k]*ny;
	yP=yvec+j; xP=xvec+j;
	mdP=mdiag+(l*n); ldP=lldiag+(l*n);
	vval=vvec[pind[k]]; sval=svec[pind[k]];
	for (i=0; i<ny; i++)
	  *(yP++)=vval*((*yP)+(*(xP++))*((*(mdP++))-(*(ldP++))))+sval;
      }
    } else {
      for (k=0; k<num; k++) {
	yP=yvec+(pind[k]*ny); xP=xvec+(pind[k]*nx);
	mdP=mdiag+(l*n); ldP=lldiag+(l*n);
	vval=vvec[pind[k]]; sval=svec[pind[k]];
	fillVec(tempP,n,0.0);
	for (i=0; i<nx; i++) {
	  j=iindx[i];
	  tempP[j]=(*(xP++))*(mdP[j]-ldP[j]);
	}
	for (i=0; i<ny; i++) {
	  j=iindy[i];
	  *(yP++)=vval*((*yP)+tempP[j])+sval;
	}
      }
    }
    mpos+=(n*szi); /* Next L^(l) */
  }

  if (ownTmp) mxFree((void*) tempP);
  mxFree((void*) iperm); mxFree((void*) pind);
  if (tickoff!=0) mxFree((void*) tickoff);
  if (indy!=0) {
    mxFree((void*) iindy);
    if (indx!=0)
      mxFree((void*) iindx);
  }
}
