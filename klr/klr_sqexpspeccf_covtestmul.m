function y = klr_sqexpspeccf_covtestmul(x,tedata)
%KLR_SQEXPSPECCF_COVTESTMUL Variant of KLR_SQEXPCF_COVTESTMUL
%  Y = KLR_SQEXPSPECCF_COVTESTMUL(X,TEDATA)
%  Variant of KLR_SQEXPCF_COVTESTMUL.

global klr klr_intern;

nn=klr.num_data; nc=klr.num_class;
nd=size(tedata,2);
if nd~=size(klr_intern.xdata,2)
  error('TEDATA has wrong size');
end
th=klr.covinfo.theta;
nte=size(tedata,1);
if klr.covinfo.tied
  temp=exp(th(1));
  wmat=temp(ones(nd,1))+th(2:(nd+1));
  y=reshape(sqexpcf(tedata,klr_intern.xdata,exp(th(nd+2)), ...
		    wmat)*reshape(x,nn,nc), ...
	    nte*nc,1);
else
  y=zeros(nte*nc,1);
  pos=1;
  for c=1:nc
    rng1=(nn*(c-1)+1):(nn*c);
    rng2=(nte*(c-1)+1):(nte*c);
    temp=exp(th(pos));
    wmat=temp(ones(nd,1))+th((pos+1):(pos+nd));
    y(rng2)=sqexpcf(tedata,klr_intern.xdata,exp(th(pos+nd+1)), ...
		    wmat)*x(rng1);
    pos=pos+nd+2;
  end
end
