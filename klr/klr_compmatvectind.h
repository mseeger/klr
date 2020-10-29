/* -------------------------------------------------------------------
 * KLR_COMPMATVECTIND / KLR_COMPMATMATIND
 *
 * Auxiliary function, does MVM Y = A(INDY,INDX)*X with indexes
 * INDY, INDX. See KLR_COMPMATVECT. N is full size of A (squared).
 * KLR_COMPMATMATIND the same, but X, Y are matrices with Q cols.
 * -------------------------------------------------------------------
 * Author: Matthias Seeger
 * ------------------------------------------------------------------- */

#ifndef COMPMATVECTIND_H
#define COMPMATVECTIND_H

#include "mex_helper.h"

void klr_compmatvectind(const double* xvec,double* yvec,const double* mat,
			int n,bool lower,const double* indy,int ny,
			const double* indx,int nx)
{
  int i,j,step1,step2,posi,posj;
  double temp,sum;
  int* indxi,*indyi;
  const double* xP;

  /* Convert indexes into INT, subtract 1 (they are base 1) */
  indyi=(int*) mxMalloc(n*sizeof(int));
  for (i=0; i<ny; i++) indyi[i]=((int) indy[i])-1;
  if (indx==0) {
    indxi=indyi; nx=ny;
  } else {
    indxi=(int*) mxMalloc(nx*sizeof(int));
    for (i=0; i<nx; i++) indxi[i]=((int) indx[i])-1;
  }
  /* Main loop */
  if (lower) {
    step1=1; step2=n;
  } else {
    step1=n; step2=1;
  }
  for (i=0; i<ny; i++) {
    posi=indyi[i];
    xP=xvec;
    for (j=0,sum=0.0; j<nx; j++) {
      if ((posj=indxi[j])<=posi)
	temp=*(mat+(step1*posi+step2*posj));
      else
	temp=*(mat+(step1*posj+step2*posi));
      sum+=temp*(*(xP++));
    }
    *(yvec++)=sum;
  }
  mxFree((void*) indyi);
  if (indx!=0) mxFree((void*) indxi);
}

void klr_compmatmatind(const double* xmat,double* ymat,int q,const double* mat,
		       int n,bool lower,const double* indy,int ny,
		       const double* indx,int nx)
{
  int i,j,step1,step2,posi,posj;
  double temp;
  int* indxi,*indyi;
  const double* xP;

  if (q==1)
    klr_compmatvectind(xmat,ymat,mat,n,lower,indy,ny,indx,nx);
  else {
    /* Convert indexes into INT, subtract 1 (they are base 1) */
    indyi=(int*) mxMalloc(n*sizeof(int));
    for (i=0; i<ny; i++) indyi[i]=((int) indy[i])-1;
    if (indx==0) {
      indxi=indyi; nx=ny;
    } else {
      indxi=(int*) mxMalloc(nx*sizeof(int));
      for (i=0; i<nx; i++) indxi[i]=((int) indx[i])-1;
    }
    /* Main loop */
    if (lower) {
      step1=1; step2=n;
    } else {
      step1=n; step2=1;
    }
    fillVec(ymat,ny*q,0.0); /* clear */
    for (i=0; i<ny; i++,ymat++) {
      posi=indyi[i];
      xP=xmat;
      for (j=0; j<nx; j++,xP++) {
	if ((posj=indxi[j])<=posi)
	  temp=*(mat+(step1*posi+step2*posj));
	else
	  temp=*(mat+(step1*posj+step2*posi));
	BLASFUNC(daxpy) (&q,&temp,xP,&nx,ymat,&ny);
      }
    }
    mxFree((void*) indyi);
    if (indx!=0) mxFree((void*) indxi);
  }
}

#endif
