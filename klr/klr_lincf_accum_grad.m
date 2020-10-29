function grad = klr_lincf_accum_grad(emat,fmat)
%KLR_LINCF_ACCUM_GRAD ACCUM_GRAD implementation for linear kernel
%  GRAD = KLR_LINCF_ACCUM_GRAD(EMAT,FMAT)
%  ACCUM_GRAD implementation, see KLR_LINCF_COVMAT for kernel
%  representation.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
nn=klr.num_data; nc=klr.num_class; n=nn*nc;
[d1 nq]=size(emat); [d2 d3]=size(fmat);
if d1~=n || d2~=n || d3~=nq
  error('EMAT, FMAT have wrong size');
end

grad=zeros(length(klr.covinfo.theta),1);
vvec=reshape(exp(klr.covinfo.theta(klr.covinfo.vpar_pos)),nc,1);
fst_reshape(fmat,nn,nc*nq);
fst_reshape(emat,nn,nc*nq);
if isfield(klr.covinfo,'spar_pos') && ~isempty(klr.covinfo.spar_pos)
  svec=reshape(exp(klr.covinfo.theta(klr.covinfo.spar_pos)),nc,1);
  % Gradient for s_p parameters
  tmat=reshape(fst_diagmul(emat,fmat,0),nc,nq);
  tvec=sum(tmat,2).*vvec;
  grad=fst_sumpos(tvec,klr.covinfo.spar_pos,length(grad));
end
% v_p parameters only
df=size(klr_intern.xdata,2);
if df<nn
  if issparse(klr_intern.xdata)
    dummy=zeros(df,nc*nq);
    fst_dspamm(dummy,klr_intern.xdata,fmat,1);
    %dummy=full(klr_intern.xdata'*fmat);
    dummy2=zeros(df,nc*nq);
    fst_dspamm(dummy2,klr_intern.xdata,emat,1);
    %dummy2=full(klr_intern.xdata'*emat);
  else
    dummy=klr_intern.xdata'*fmat;
    dummy2=klr_intern.xdata'*emat;
  end
  tmat=reshape(fst_diagmul(dummy,dummy2,0),nc,nq);
  tvec=sum(tmat,2).*vvec;
else
  if issparse(klr_intern.xdata)
    dummy=zeros(nn,nc*nq);
    fst_dspammt(dummy,klr_intern.xdata,fmat);
    %dummy=full(klr_intern.xdata*full(klr_intern.xdata'*fmat));
  else
    dummy=klr_intern.xdata*(klr_intern.xdata'*fmat);
  end
  tmat=reshape(fst_diagmul(emat,dummy,0),nc,nq);
  tvec=sum(tmat,2).*vvec;
end
grad=grad+fst_sumpos(tvec,klr.covinfo.vpar_pos,length(grad));
