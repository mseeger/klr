function grad = klr_rbfcf_accum_grad(emat,fmat)
%KLR_RBFCF_ACCUM_GRAD ACCUM_GRAD implementation for RBF kernel
%  GRAD = KLR_RBFCF_ACCUM_GRAD(EMAT,FMAT)
%  ACCUM_GRAD implementation for RBF kernel. See
%  KLR_RBFCF_COMP_PREC for precomp. matrix.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
nn=klr.num_data; nc=klr.num_class; n=nn*nc;
[d1 nq]=size(emat); [d2 d3]=size(fmat);
if d1~=n || d2~=n || d3~=nq
  error('EMAT, FMAT have wrong size');
end
if isfield(klr.covinfo,'fixvar') && klr.covinfo.fixvar
  fixvar=1;
  sz=1;
else
  fixvar=0;
  sz=2;
end
if ~klr.covinfo.tied
  sz=sz*nc;
end
thpos=1;
if isfield(klr.covinfo,'thpos')
  thpos=klr.covinfo.thpos;
end
theta=klr.covinfo.theta(thpos:(thpos+sz-1));
grad=zeros(sz,1);
if ~klr.covinfo.tied
  dgind=(0:(nn-1))'*(nn+1)+1;
  tmat=zeros(nn,nq);
  if ~fixvar
    % C parameters
    actc=1; uplo='L ';
    for c=1:nc
      rng=(nn*(c-1)+1):(nn*c);
      klr_intern.covmat{actc}(dgind)=klr.covdiag(rng);
      posv=[rng(1); 1; nn; nq];
      fst_dsymm(tmat,{klr_intern.covmat{actc}; [1; 1; nn; nn]; uplo}, ...
		{emat; posv});
      grad(2*c)=sum(fst_diagmul(tmat,{fmat; posv},0));
      if uplo(1)=='L'
	uplo(1)='U';
      else
	uplo(1)='L';
	actc=actc+1;
      end
    end
  end
  % W parameters
  pos=1; actc=0; uplo='U ';
  if ~fixvar
    incpos=2;
  else
    incpos=1;
  end
  for c=1:nc
    if uplo(1)=='L'
      uplo(1)='U';
    else
      uplo(1)='L';
      actc=actc+1;
      % Diag. of KLR_INTERN.PRECMAT is 0, no need to fix kernel
      % matrix diagonal
      tmat2=klr_intern.covmat{actc}.*klr_intern.precmat;
    end
    rng=(nn*(c-1)+1):(nn*c);
    posv=[rng(1); 1; nn; nq];
    fst_dsymm(tmat,{tmat2; [1; 1; nn; nn]; uplo},{emat; posv});
    grad(pos)=exp(theta(pos))*sum(fst_diagmul(tmat,{fmat; posv}, ...
					      0));
    pos=pos+incpos;
  end
else
  tmat=zeros(nn,nc*nq);
  fst_reshape(emat,nn,nc*nq);
  fst_reshape(fmat,nn,nc*nq);
  if ~fixvar
    % C parameters
    fst_dsymm(tmat,{klr_intern.covmat; [1; 1; nn; nn]; 'L '},emat);
    grad(2)=sum(fst_diagmul(tmat,fmat,0));
  end
  % W parameter
  fst_dsymm(tmat,{klr_intern.covmat.*klr_intern.precmat; [1; 1; nn; ...
		    nn]; 'L '},emat);
  grad(1)=exp(theta(1))*sum(fst_diagmul(tmat,fmat,0));
  fst_reshape(emat,n,nq);
  fst_reshape(fmat,n,nq);
end
