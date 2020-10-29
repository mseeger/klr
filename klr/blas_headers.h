/* -------------------------------------------------------------------
 * BLAS headers
 * -------------------------------------------------------------------
 * Author: Matthias Seeger
 * ------------------------------------------------------------------- */

#ifndef BLAS_HEADER_H
#define BLAS_HEADER_H

extern void BLASFUNC(dswap) (int* n,double* x,int* incx,double* y,int* incy);

extern void BLASFUNC(dcopy) (int* n,const double* x,int* incx,double* y,
			     int* incy);

extern void BLASFUNC(dscal) (int* n,const double* alpha,double* x,int* incx);

extern double BLASFUNC(ddot) (int* n,const double* a,int* lda,const double* b,
			      int* ldb);

extern void BLASFUNC(daxpy) (int *n,const double* alpha,const double* x,
			     int* incx,double* y,int* incy);

extern void BLASFUNC(dsymv) (const char* uplo,int* n,double* alpha,
			     const double* a,int* lda,const double* x,
			     int* incx,double* beta,double* y,int* incy);

extern void BLASFUNC(dgemm) (const char* tra,const char* trb,int* m,int *n,
			     int* k,double* alpha,const double* a,int* lda,
			     const double* b,int* ldb,double* beta,
			     double* c,int* ldc);

extern void BLASFUNC(dsymm) (const char* side,const char* uplo,int* m,int* n,
			     double* alpha,const double* a,int* lda,
			     const double* b,int* ldb,double* beta,double* c,
			     int* ldc);

extern void BLASFUNC(dtrsm) (const char* side,const char* uplo,
			     const char* trans,const char* diag,int* m,int* n,
			     double* alpha,const double* a,int* lda,
			     const double* b,int* ldb);

extern void BLASFUNC(dtrmm) (const char* side,const char* uplo,
			     const char* trans,const char* diag,int* m,int* n,
			     double* alpha,const double* a,int* lda,
			     const double* b,int* ldb);

#endif
