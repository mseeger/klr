function y = klr_lincf_covpartmul(x,iind,jind)
%KLR_LINCF_COVPARTMUL COVPARTMUL implementation for linear kernel
%  Y = KLR_LINCF_COVPARTMUL(X,IIND,JIND)
%  Implements COVPARMUL. See KLR_LINCF_COVMUL for kernel
%  representation.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
ni=length(iind); nj=length(jind);
nc=klr.num_class;
if length(x)~=nj*nc
  error('X has wrong size');
end

fst_reshape(x,nj,nc);
if issparse(klr_intern.xdata)
  y=zeros(ni,nc);
  fst_dspammt2(y,klr_intern.xdata(iind,:), ...
	       klr_intern.xdata(jind,:),x);
else
  y=klr_intern.xdata(iind,:)*(klr_intern.xdata(jind,:)'*x);
end
vvec=exp(klr.covinfo.theta(klr.covinfo.vpar_pos));
fst_muldiag(y,vvec,0);
if isfield(klr.covinfo,'spar_pos') && ...
      ~isempty(klr.covinfo.spar_pos)
  svec=exp(klr.covinfo.theta(klr.covinfo.spar_pos));
  fst_addvec(y,svec,0);
end
fst_reshape(x,nj*nc,1);
fst_reshape(y,ni*nc,1);
