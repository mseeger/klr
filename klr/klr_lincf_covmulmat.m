function y = klr_lincf_covmulmat(x)
%KLR_LINCF_COVMULMAT COVMULMAT implementation for linear kernel
%  Y = KLR_LINCF_COVMULMAT(X)
%  Implements COVMULMAT. See KLR_LINCF_COVMUL for kernel
%  representation.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
nn=klr.num_data; nc=klr.num_class;
if size(x,1)~=nn*nc
  error('X has wrong size');
end
nq=size(x,2);

fst_reshape(x,nn,nc*nq);
if issparse(klr_intern.xdata)
  y=zeros(nn,nc*nq);
  fst_dspammt(y,klr_intern.xdata,x);
else
  y=klr_intern.xdata*(klr_intern.xdata'*x);
end
vvec=reshape(exp(klr.covinfo.theta(klr.covinfo.vpar_pos)),nc,1);
tvec=reshape(vvec(:,ones(nq,1)),nc*nq,1);
fst_muldiag(y,tvec,0);
if isfield(klr.covinfo,'spar_pos') && ...
      ~isempty(klr.covinfo.spar_pos)
  svec=reshape(exp(klr.covinfo.theta(klr.covinfo.spar_pos)),nc,1);
  tvec=reshape(svec(:,ones(nq,1)),nc*nq,1);
  fst_addvec(y,tvec,0);
end
fst_reshape(x,nn*nc,nq);
fst_reshape(y,nn*nc,nq);
