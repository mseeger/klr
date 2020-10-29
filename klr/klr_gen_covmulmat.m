function y = klr_gen_covmulmat(x)
%KLR_GEN_COVMULMAT Generic COVMULMAT implementation
%  Y = KLR_GEN_COVMULMAT(X)
%  Computes Y = K*X for the kernel matrix K. K is block-diagonal,
%  the blocks are obtained from KLR_INTERN.COVMAT (see
%  KLR_GEN_COVMUL). X is a matrix.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
nn=klr.num_data; nc=klr.num_class; n=nn*nc;
nq=size(x,2);
if size(x,1)~=n
  error('X has wrong size');
end
if ~klr.covinfo.tied
  y=zeros(n,nq);
  dgind=(0:(nn-1))'*(nn+1)+1;
  actc=1; uplo='L ';
  for c=1:nc
    rng=(nn*(c-1)+1):(nn*c);
    klr_intern.covmat{actc}(dgind)=klr.covdiag(rng);
    posv=[rng(1); 1; nn; nq];
    fst_dsymm({y; posv},{klr_intern.covmat{actc}; [1; 1; nn; nn]; ...
		    uplo},{x; posv});
    if uplo(1)=='L'
      uplo(1)='U';
    else
       uplo(1)='L';
       actc=actc+1;
    end
  end
else
  fst_reshape(x,nn,nc*nq);
  y=zeros(nn,nc*nq);
  fst_dsymm(y,{klr_intern.covmat{actc}; [1; 1; nn; nn]; 'L '},x);
  fst_reshape(x,n,nq);
  fst_reshape(y,n,nq);
end
