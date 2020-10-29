function dummy = klr_lincf_comp_prec(varargin)
%KLR_LINCF_COMP_PREC COMP_PREC implementation for linear kernel
%  KLR_LINCF_COMP_PREC({NEWDATA=0})
%  Nothing to do for the linear kernel, but we check that the data
%  matrix is given in KLR_INTERN.XDATA.

global klr klr_intern;

nn=klr.num_data; nc=klr.num_class;
if ~isfield(klr_intern,'xdata') || nn~=size(klr_intern.xdata,1)
  error('Need data matrix in KLR_INTERN.XDATA');
end
tvec=full(sum(klr_intern.xdata.*klr_intern.xdata,2));
klr.covdiag=tvec(:,ones(nc,1));
vvec=exp(klr.covinfo.theta(klr.covinfo.vpar_pos));
fst_muldiag(klr.covdiag,vvec,0);
if isfield(klr.covinfo,'spar_pos') && ...
      ~isempty(klr.covinfo.spar_pos)
  svec=exp(klr.covinfo.theta(klr.covinfo.spar_pos));
  fst_addvec(klr.covdiag,svec,0);
end
klr.covdiag=reshape(klr.covdiag,nn*nc,1);
klr.covinfo.prec_ok=1;
% DEBUG!!!
%if klr.covinfo.tied
%  error('UURG');
%end
%tmat=klr_intern.xdata*klr_intern.xdata';
%for i=1:floor(nc/2)
%  klr_intern.covmat{i}=tril(vvec(2*i-1)*tmat,-1)+triu(vvec(2*i)* ...
%						  tmat,1);
%end
%if mod(nc,2)==1
%  klr_intern.covmat{ceil(nc/2)}=vvec(nc)*tmat;
%end
