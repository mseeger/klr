function y = klr_wrap_covpartmul(x,iind,jind)
%KLR_WRAP_COVPARTMUL Wrapper for COVPARTMUL calls
%  Y = KLR_WRAP_COVPARTMUL(X,IIND,JIND)
%  Wrapper for calls to COVPARTMUL. Same functionality as
%  KLR_WRAP_COVMUL for the COVMUL primitive.

global klr klr_intern;

if ~isfield(klr,'mixmat') || ~isfield(klr.mixmat,'use') || ...
      ~klr.mixmat.use
  % Standard case
  y=feval(klr.covpartmul,x,iind,jind);
else
  % Mixing matrix B
  nc=klr.num_class;
  if klr.mixmat.use==2
    i=klr.mixmat.thpos;
    klr.mixmat.bmat=reshape(klr.covinfo.theta(i:(i+nc*nc-1)),nc, ...
			    nc);
  end
  bmat=klr.mixmat.bmat;
  ni=length(iind); nj=length(jind);
  nc=klr.num_class;
  if length(x)~=nj*nc
    error('X has wrong size');
  end
  tvec=reshape(reshape(x,nj,nc)*bmat,nj*nc,1);
  tvec=feval(klr.covpartmul,tvec,iind,jind);
  y=reshape(reshape(tvec,ni,nc)*bmat',ni*nc,1);
end
