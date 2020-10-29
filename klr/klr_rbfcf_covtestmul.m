function y = klr_rbfcf_covtestmul(x,tedata)
%KLR_RBFCF_COVTESTMUL COVTESTMUL implementation for RBF kernel
%  Y = KLR_RBFCF_COVTESTMUL(X,TEDATA)
%  Computes Y = K*X, K the kernel matrix between test data TEDATA
%  and training data KLR_INTERN.XDATA, kernel parameters from
%  KLR.COVINFO.THETA. No precomputation required.

global klr klr_intern;

nn=klr.num_data; nc=klr.num_class;
if size(tedata,2)~=size(klr_intern.xdata,2)
  size(tedata)
  size(klr_intern.xdata)
  error('TEDATA has wrong size');
end
if isfield(klr.covinfo,'fixvar') && klr.covinfo.fixvar
  fixvar=1;
else
  fixvar=0;
end
thpos=1;
if isfield(klr.covinfo,'thpos')
  thpos=klr.covinfo.thpos;
end
th=klr.covinfo.theta(thpos:end);
nte=size(tedata,1);
lvar=0;
if klr.covinfo.tied
  if ~fixvar
    lvar=th(2);
  end
  y=reshape(radialcf(tedata,klr_intern.xdata,exp(lvar),exp(th(1)))* ...
	    reshape(x,nn,nc),nte*nc,1);
else
  y=zeros(nte*nc,1);
  for c=1:nc
    rng1=(nn*(c-1)+1):(nn*c);
    rng2=(nte*(c-1)+1):(nte*c);
    if ~fixvar
      lvar=th(2*c);
    end
    y(rng2)=radialcf(tedata,klr_intern.xdata,exp(lvar),exp(th(2* ...
						  c-1)))*x(rng1);
  end
end
