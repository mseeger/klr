function grad = klr_sqexpcf_accum_grad(emat,fmat)
%KLR_SQEXPCF_ACCUM_GRAD ACCUM_GRAD implementation for SQEXP kernel
%  GRAD = KLR_SQEXPCF_ACCUM_GRAD(EMAT,FMAT)
%  Computes gradient for SQEXP kernel. The parameters THETA are
%  read from KLR.COVINFO.THETA. GRAD(I) is computed as trace of
%  EMAT'*((d K)/(d THETA(I)))*FMAT. See KLR_SQEXPCF_COMP_PREC.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
nn=klr.num_data; nc=klr.num_class; n=nn*nc;
nd=size(klr_intern.xdata,2);
[d1 nq]=size(emat); [d2 d3]=size(fmat);
if d1~=n || d2~=n || d3~=nq
  error('EMAT, FMAT have wrong size');
end
if isfield(klr.covinfo,'fixvar') && klr.covinfo.fixvar
  fixvar=1;
  sz=nd;
else
  fixvar=0;
  sz=nd+1;
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
      grad((nd+1)*c)=sum(fst_diagmul(tmat,{fmat; posv},0));
      if uplo(1)=='L'
	uplo(1)='U';
      else
	uplo(1)='L';
	actc=actc+1;
      end
    end
  end
  % W parameters
  pos=1; actc=1; uplo='L ';
  if ~fixvar
    incpos=nd+1;
  else
    incpos=nd;
  end
  for c=1:nc
    rng=(nn*(c-1)+1):(nn*c);
    posv=[rng(1); 1; nn; nq];
    klr_intern.covmat{actc}(dgind)=klr.covdiag(rng);
    off=(c-1)*incpos;
    for d=1:nd
      temp=full(klr_intern.xdata(:,d));
      tmat2=emat(rng,:);
      fst_muldiag(tmat2,temp,1);
      fst_dsymm(tmat,{klr_intern.covmat{actc}; [1; 1; nn; nn]; uplo}, ...
		tmat2);
      tmat2=fmat(rng,:);
      fst_muldiag(tmat2,temp,1);
      tscal=sum(fst_diagmul(tmat,tmat2,0));
      temp=temp.*temp;
      tmat2=emat(rng,:);
      fst_muldiag(tmat2,temp,1);
      fst_dsymm(tmat,{klr_intern.covmat{actc}; [1; 1; nn; nn]; uplo}, ...
		tmat2);
      tscal=tscal-0.5*sum(fst_diagmul(tmat,{fmat; posv},0));
      tmat2=fmat(rng,:);
      fst_muldiag(tmat2,temp,1);
      fst_dsymm(tmat,{klr_intern.covmat{actc}; [1; 1; nn; nn]; uplo}, ...
		{emat; posv});
      grad(off+d)=(exp(theta(off+d))/nd)* ...
	  (tscal-0.5*sum(fst_diagmul(tmat,tmat2,0)));
    end
    if uplo(1)=='L'
      uplo(1)='U';
    else
      uplo(1)='L';
      actc=actc+1;
    end
  end
else
  tmat=zeros(nn,nc*nq);
  fst_reshape(emat,nn,nc*nq);
  fst_reshape(fmat,nn,nc*nq);
  if ~fixvar
    % C parameter
    fst_dsymm(tmat,{klr_intern.covmat; [1; 1; nn; nn]; 'L '},emat);
    grad(nd+1)=sum(fst_diagmul(tmat,fmat,0));
  end
  % W parameter
  for d=1:nd
    temp=full(klr_intern.xdata(:,d));
    tmat2=emat(:,:); % force copy
    fst_muldiag(tmat2,temp,1);
    fst_dsymm(tmat,{klr_intern.covmat; [1; 1; nn; nn]; 'L '}, ...
	      tmat2);
    tmat2=fmat(:,:); % force copy
    fst_muldiag(tmat2,temp,1);
    tscal=sum(fst_diagmul(tmat,tmat2,0));
    temp=temp.*temp;
    tmat2=emat(:,:); % force copy
    fst_muldiag(tmat2,temp,1);
    fst_dsymm(tmat,{klr_intern.covmat; [1; 1; nn; nn]; 'L '}, ...
	      tmat2);
    tscal=tscal-0.5*sum(fst_diagmul(tmat,fmat,0));
    tmat2=fmat(:,:); % force copy
    fst_muldiag(tmat2,temp,1);
    fst_dsymm(tmat,{klr_intern.covmat; [1; 1; nn; nn]; 'L '}, ...
	      emat);
    grad(d)=(exp(theta(d))/nd)*(tscal-0.5*sum(fst_diagmul(tmat, ...
						  tmat2,0)));
  end
  fst_reshape(emat,nn*nc,nq);
  fst_reshape(fmat,nn*nc,nq);
end
