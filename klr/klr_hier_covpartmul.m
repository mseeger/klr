function y = klr_hier_covpartmul(x,iind,jind)
%KLR_HIER_COVPARTMUL COVPARTMUL implementation for hierarchical classif.
%  Y = KLR_HIER_COVPARTMUL(X,IIND,JIND)
%  Implements COVPARMUL. See KLR_HIER_COVMUL for format of kernel
%  matrix blocks.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

if ~isfield(klr,'hierarch') || ~klr.hierarch.use
  error('Hierarchical classification mode not active');
end
if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
ni=length(iind); nj=length(jind);
nc=klr.num_class; np=klr.hierarch.nump;
if length(x)~=nj*nc
  error('X has wrong size');
end
% Multiplication by PHI'
xtemp=zeros(nj,np);
xtemp(:,klr.hierarch.leafs)=reshape(x,nj,nc);
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
  if issparse(klr_intern.xdata)
    dummy=zeros(ni,np);
    fst_dspammt2(dummy,klr_intern.xdata(iind,:), ...
		 klr_intern.xdata(jind,:),xtemp);
  else
    dummy=klr_intern.xdata(iind,:)* ...
	  (klr_intern.xdata(jind,:)'*xtemp);
  end
  fst_muldiag(dummy,vvec,0);
  if use_spars
    fst_addvec(dummy,svec,0);
  end
else
  fst_reshape(xtemp,nj*np,1);
  if ~isfield(klr.hierarch,'lowrk') || ~klr.hierarch.lowrk.use
    dummy=klr_compkernmatvect(klr.num_data,xtemp,vvec,svec, ...
			      klr_intern.covmat, ...
			      klr_intern.covtldiag, ...
			      klr.hierarch.l2p,iind,jind, ...
			      klr_intern.covmul_tmpbuff);
  else
    dummy=klr_compkernmatvect_lr(klr.num_data,xtemp,vvec,svec, ...
				 klr_intern.covmat, ...
				 klr_intern.covtldiag, ...
				 klr.hierarch.lowrk.cmat_diag, ...
				 klr.hierarch.lowrk.perm, ...
				 klr.hierarch.lowrk.actsz, ...
				 klr.hierarch.l2p,iind,jind, ...
				 klr_intern.covmul_tmpbuff);
  end
  fst_reshape(dummy,ni,np);
end
% Multiplication by PHI
klr_compmatphit(dummy,klr.hierarch.childn);
y=reshape(dummy(:,klr.hierarch.leafs),ni*nc,1);
