function grad = klr_wrap_accum_grad(emat,fmat)
%KLR_WRAP_ACCUM_GRAD Wrapper for ACCUM_GRAD calls
%  GRAD = KLR_WRAP_ACCUM_GRAD(EMAT,FMAT)
%  Wrapper for calls to ACCUM_GRAD, dealing with mixing matrix.
%  If no mixing matrix is used, just calls KLR.ACCUM_GRAD.
%  Otherwise, assumes that mixing matrix BMAT is kept in trailing
%  entires of KLR.COVINFO.THETA, and KLR.ACCUM_GRAD deals with
%  the prefix.

global klr klr_intern;

if isfield(klr,'mixmat') && isfield(klr.mixmat,'use')
  usemix=klr.mixmat.use;
else
  usemix=0;
end
nc=klr.num_class; nn=klr.num_data; n=nn*nc;
nq=size(emat,2);
if size(emat,1)~=n || size(fmat,1)~=n || size(fmat,2)~=nq
  error('EMAT, FMAT wrong');
end
if ~usemix
  % Other kernel parameters
  grad=feval(klr.accum_grad,emat,fmat);
else
  if usemix==2
    if isfield(klr.covinfo,'thpos') && klr.covinfo.thpos~=1
      error('Kernel parameters must be prefix in KLR.COVINFO.THETA');
    end
    i=klr.mixmat.thpos;
    if i+nc*nc-1~=length(klr.covinfo.theta)
      error('Mixing matrix must be postfix in KLR.COVINFO.THETA');
    end
    klr.mixmat.bmat=reshape(klr.covinfo.theta(i:end),nc,nc);
  end
  % Premultiply EMAT, FMAT
  bmat=klr.mixmat.bmat;
  etmat=reshape(permute(reshape(reshape(permute(reshape(emat,nn,nc,nq),[1 ...
		    3 2]),nn*nq,nc)*bmat,nn,nq,nc),[1 3 2]),n,nq);
  ftmat=reshape(permute(reshape(reshape(permute(reshape(fmat,nn,nc,nq),[1 ...
		    3 2]),nn*nq,nc)*bmat,nn,nq,nc),[1 3 2]),n,nq);
  % Other kernel pars
  grad=feval(klr.accum_grad,etmat,ftmat);
  if usemix==2
    % Gradient w.r.t. BMAT
    etmat=reshape(permute(reshape(feval(klr.covmulmat,etmat),nn,nc,nq),[1 ...
		    3 2]),nn*nq,nc);
    ftmat=reshape(permute(reshape(feval(klr.covmulmat,ftmat),nn,nc,nq),[1 ...
		    3 2]),nn*nq,nc);
    tmat=reshape(permute(reshape(fmat,nn,nc,nq),[1 3 2]),nn*nq,nc)'* ...
	 etmat+reshape(permute(reshape(emat,nn,nc,nq),[1 3 2]),nn*nq, ...
		       nc)'*ftmat;
    grad=[grad; reshape(tmat,nc*nc,1)];
  end
end
