function y = klr_sqexpcf_covtestmul(x,tedata)
%KLR_SQEXPCF_COVTESTMUL COVTESTMUL implementation for SQEXP kernel
%  Y = KLR_SQEXPCF_COVTESTMUL(X,TEDATA)
%  Computes Y = K*X, K the kernel matrix between test data TEDATA
%  and training data KLR_INTERN.XDATA, kernel parameters from
%  KLR.COVINFO.THETA. No precomputation required.

global klr klr_intern;

nn=klr.num_data; nc=klr.num_class;
nd=size(tedata,2);
if nd~=size(klr_intern.xdata,2)
  error('TEDATA has wrong size');
end
if isfield(klr.covinfo,'fixvar') & klr.covinfo.fixvar
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
var=1;
if klr.covinfo.tied
  if ~fixvar
    var=exp(th(nd+1));
  end
  y=reshape(sqexpcf(tedata,klr_intern.xdata,var, ...
		    exp(th(1:nd)))*reshape(x,nn,nc), ...
	    nte*nc,1);
else
  y=zeros(nte*nc,1);
  pos=1;
  if ~fixvar
    incpos=nd+1;
  else
    incpos=nd;
  end
  for c=1:nc
    if ~fixvar
      var=exp(th(pos+nd));
    end
    rng1=(nn*(c-1)+1):(nn*c);
    rng2=(nte*(c-1)+1):(nte*c);
    y(rng2)=sqexpcf(tedata,klr_intern.xdata,var, ...
		    exp(th(pos:(pos+nd-1))))*x(rng1);
    pos=pos+incpos;
  end
end
