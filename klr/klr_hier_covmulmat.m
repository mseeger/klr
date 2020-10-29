function y = klr_hier_covmulmat(x)
%KLR_HIER_COVMULMAT COVMULMAT implementation for hierarchical classif.
%  Y = KLR_HIER_COVMULMAT(X)
%  Computes Y = K*X for the kernel matrix K. See KLR_HIER_COVMUL
%  for representation of K. X is a matrix.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

if ~isfield(klr,'hierarch') || ~klr.hierarch.use
  error('Hierarchical classification mode not active');
end
if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
nn=klr.num_data; nc=klr.num_class;
np=klr.hierarch.nump;
nq=size(x,2);
if size(x,1)~=nn*nc
  error('X has wrong size');
end
% Multiplication by PHI'
xtemp=zeros(nn*nq,np);
dummy=x(:,:); % force true copy
fst_flipdims(dummy,nn);
xtemp(:,klr.hierarch.leafs)=dummy;
klr_compmatphi(xtemp,klr.hierarch.childn);
% Multiplication by block-diag. tilde kernel matrix
vvec=exp(klr.covinfo.theta(klr.covinfo.vpar_pos));
use_spars=isfield(klr.covinfo,'spar_pos') && ...
	  ~isempty(klr.covinfo.spar_pos);
if use_spars
  svec=exp(klr.covinfo.theta(klr.covinfo.spar_pos));
else
  svec=zeros(np,1);
end
if isfield(klr.hierarch,'linear_kern') && klr.hierarch.linear_kern
  % Linear kernel
  fst_reshape(xtemp,nn,nq*np);
  if issparse(klr_intern.xdata)
    df=size(klr_intern.xdata,2);
    if df<nn
      dummy=zeros(df,nq*np);
      fst_dspamm(dummy,klr_intern.xdata,xtemp,1);
      fst_dspamm(xtemp,klr_intern.xdata,dummy,0);
    else
      dummy=zeros(nn,nq*np);
      fst_dspammt(dummy,klr_intern.xdata,xtemp);
      xtemp=dummy(:,:); % force copy
    end
  else
    xtemp=klr_intern.xdata*(klr_intern.xdata'*xtemp);
  end
  fst_reshape(xtemp,nn*nq,np);
  fst_muldiag(xtemp,vvec,0);
  if use_spars
    fst_addvec(xtemp,svec,0);
  end
else
  fst_flipdims(xtemp,nn);
  if ~isfield(klr.hierarch,'lowrk') || ~klr.hierarch.lowrk.use
    xtemp=klr_compkernmatmat(nn,xtemp,vvec,svec,klr_intern.covmat, ...
			     klr_intern.covtldiag,klr.hierarch.l2p, ...
			     nn,klr_intern.covmul_tmpbuff);
  else
    xtemp=klr_compkernmatmat_lr(nn,xtemp,vvec,svec,klr_intern.covmat, ...
				klr_intern.covtldiag, ...
				klr.hierarch.lowrk.cmat_diag, ...
				klr.hierarch.lowrk.perm, ...
				klr.hierarch.lowrk.actsz, ...
				klr.hierarch.l2p,nn, ...
				klr_intern.covmul_tmpbuff);
  end
  fst_flipdims(xtemp,nn);
end
% Multiplication by PHI
klr_compmatphit(xtemp,klr.hierarch.childn);
y=xtemp(:,klr.hierarch.leafs);
fst_flipdims(y,nn);
