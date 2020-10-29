function y = klr_wrap_covmul(x,varargin)
%KLR_WRAP_COVMUL Wrapper for COVMUL calls
%  Y = KLR_WRAP_COVMUL(X,{IND})
%  Wrapper for calls to COVMUL, Y = K*X for kernel matrix K.
%  In the simplest setting, we just call the function with handle
%  KLR.COVMUL.
%  If KLR.MIXMAT exists and KLR.MIXMAT.USE==1, a mixing matrix is
%  used whose values are stored in KLR.COVINFO.THETA starting from
%  position KLR.MIXMAT.THPOS. In this case, the application of
%  KLR.COVMUL is wrapped left and right by multiplications with the
%  mixing matrix.

global klr klr_intern;

if ~isfield(klr,'mixmat') || ~isfield(klr.mixmat,'use') || ...
      ~klr.mixmat.use
  % Standard case
  y=feval(klr.covmul,x,varargin{:});
else
  % Mixing matrix B
  nc=klr.num_class;
  if klr.mixmat.use==2
    i=klr.mixmat.thpos;
    klr.mixmat.bmat=reshape(klr.covinfo.theta(i:(i+nc*nc-1)),nc, ...
			    nc);
  end
  bmat=klr.mixmat.bmat;
  n=size(x,1);
  if mod(n,nc)~=0
    error('X has wrong size');
  end
  nn=n/nc;
  tvec=reshape(reshape(x,nn,nc)*bmat,n,1);
  tvec=feval(klr.covmul,tvec,varargin{:});
  y=reshape(reshape(tvec,nn,nc)*bmat',n,1);
end
