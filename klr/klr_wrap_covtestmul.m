function y = klr_wrap_covtestmul(x,tedata)
%KLR_WRAP_COVTESTMUL Wrapper for COVTESTMUL calls
%  Y = KLR_WRAP_COVTESTMUL(X,TEDATA)
%  Wrapper for calls to COVPARTMUL. Same functionality as
%  KLR_WRAP_COVMUL for the COVMUL primitive.

global klr klr_intern;

if ~isfield(klr,'mixmat') || ~isfield(klr.mixmat,'use') || ...
      ~klr.mixmat.use
  % Standard case
  y=feval(klr.covtestmul,x,tedata);
else
  % Mixing matrix B
  nc=klr.num_class;
  if klr.mixmat.use==2
    i=klr.mixmat.thpos;
    klr.mixmat.bmat=reshape(klr.covinfo.theta(i:(i+nc*nc-1)),nc, ...
			    nc);
  end
  bmat=klr.mixmat.bmat;
  nn=klr.num_data; nte=size(tedata,1);
  nc=klr.num_class;
  if length(x)~=nn*nc
    error('X has wrong size');
  end
  tvec=feval(klr.covtestmul,reshape(reshape(x,nn,nc)*bmat,nn*nc,1), ...
	     tedata);
  y=reshape(reshape(tvec,nte,nc)*bmat',nte*nc,1);
end
