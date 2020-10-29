function y = klr_lincf_covshuffle(ind,cind,flag)
%KLR_LINCF_COVSHUFFLE COVSHUFFLE implementation for linear kernel
%  Y = KLR_LINCF_COVSHUFFLE(IND,CIND,FLAG)
%  Implements COVSHUFFLE, see KLR_LINCF_COVMAT for kernel
%  representation.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
if size(ind,2)~=1
  ind=ind';
end;
if size(cind,2)~=1
  cind=cind';
end;
nc=klr.num_class; nn=klr.num_data;
fi=[ind; cind];
t1=nn*(0:(nc-1));
tind=reshape(fi(:,ones(nc,1))+t1(ones(nn,1),:),nn*nc,1);
if flag
  klr_intern.xdata=klr_intern.xdata(fi,:);
  klr.covdiag=klr.covdiag(tind);
else
  klr_intern.xdata(fi,:)=klr_intern.xdata;
  klr.covdiag(tind)=klr.covdiag;
end
