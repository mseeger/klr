function dummy = klr_sqexpspeccf_comp_prec
%KLR_SQEXPSPECCF_COMP_PREC Variant of KLR_SQEXPCF_COMP_PREC
%  KLR_SQEXPSPECCF_COMP_PREC
%  Same as KLR_SQEXPCF_COMP_PREC, but the W diag. matrix is of the
%  form w_0 I + V, V diagonal. THETA has the form
%  [log(w_0); V; log(C)].
%  The data matrix (cases as rows) must be given in
%  KLR_INTERN.XDATA.

global klr klr_intern;

nn=klr.num_data; nc=klr.num_class;
if nn~=size(klr_intern.xdata,1)
  error('KLR.NUM_DATA or KLR_INTERN.XDATA wrong');
end
theta=klr.covinfo.theta;
nd=size(klr_intern.xdata,2); sz=nd+2;
if ~klr.covinfo.tied
  sz=sz*nc;
end
if length(theta)~=sz
  error('Size of KLR.COVINFO.THETA wrong');
end
if klr.verbose>1
  fprintf(1,'KLR_SQEXPSPECCF_COMP_PREC: Doing precomputation.\n');
end
clear klr_intern.covmat;
temp=exp(theta(1));
wmat=temp(ones(nd,1))+theta(2:(nd+1));
klr_intern.covmat{1}=sqexpcf(klr_intern.xdata,klr_intern.xdata, ...
			     exp(theta(nd+2)),wmat);
if ~klr.covinfo.tied
  pos=nd+2;
  for c=2:nc
    temp=exp(theta(pos));
    wmat=temp(ones(nd,1))+theta((pos+1):(pos+nd));
    klr_intern.covmat{c}=sqexpcf(klr_intern.xdata,klr_intern.xdata, ...
				 exp(theta(pos+nd+1)),wmat);
    pos=pos+nd+2;
  end
end
if klr.covinfo.tied
  temp=exp(theta(nd+2));
  klr.covdiag=temp(ones(nn*nc,1));
else
  temp=exp(theta((nd+2)*(1:nc)));
  if size(temp,2)==1
    temp=temp';
  end
  klr.covdiag=reshape(temp(ones(nn,1),:),nn*nc,1);
end
klr.covinfo.prec_ok=1;
