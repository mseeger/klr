function y = klr_hier_covmul(x,varargin)
%KLR_HIER_COVMUL COVMUL implementation for hierarchical classif.
%  Y = KLR_HIER_COVMUL(X,{IND})
%  Computes Y = K*X for the kernel matrix K. See docum. for
%  hierarch. classif. for how K is structured and in which
%  variables it is maintained.
%  If IND is given and not empty, it is an index selecting
%  submatrices of the kernel matrix blocks. X, Y are smaller in
%  this case.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

if ~isfield(klr,'hierarch') || ~klr.hierarch.use
  error('Hierarchical classification mode not active');
end
if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
ind=[]; nn=klr.num_data;
if isfield(klr,'worksubind')
  ind=klr.worksubind;
  nn=length(ind);
end
if nargin>1
  temp=varargin{1};
  if ~isempty(temp)
    if ~isempty(ind)
      error('Cannot use argument IND together with KLR.WORKSUBIND');
    end
    ind=temp;
    if ind(1)==0
      if length(ind)~=2
	error('Invalid IND');
      end
      nn=ind(2);
      if nn<1 || nn>klr.num_data
	error('Invalid IND');
      end
    else
      [nn,b]=size(ind);
      if b~=1
	ind=ind';
	nn=b;
      end
    end
  end
end
nc=klr.num_class; np=klr.hierarch.nump;
if length(x)~=nn*nc
  error('X has wrong size');
end
% Multiplication by PHI'
xtemp=zeros(nn,np);
xtemp(:,klr.hierarch.leafs)=reshape(x,nn,nc);
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
    df=size(klr_intern.xdata,2);
    if df<nn
      dummy=zeros(df,np);
      if isempty(ind)
	fst_dspamm(dummy,klr_intern.xdata,xtemp,1);
	fst_dspamm(xtemp,klr_intern.xdata,dummy,0);
	%xtemp=full(klr_intern.xdata*full(klr_intern.xdata'*xtemp));
      elseif ind(1)==0
	spx=klr_intern.xdata(1:nn,:);
	fst_dspamm(dummy,spx,xtemp,1);
	fst_dspamm(xtemp,spx,dummy,0);
	%xtemp=full(klr_intern.xdata(1:nn,:)* ...
	%	 full(klr_intern.xdata(1:nn, :)'*xtemp));
      else
	spx=klr_intern.xdata(ind,:);
	fst_dspamm(dummy,spx,xtemp,1);
	fst_dspamm(xtemp,spx,dummy,0);
	%xtemp=full(klr_intern.xdata(ind,:)* ...
	%	 full(klr_intern.xdata(ind,:)'*xtemp));
      end
    else
      dummy=zeros(nn,np);
      if isempty(ind)
	fst_dspammt(dummy,klr_intern.xdata,xtemp);
      elseif ind(1)==0
	fst_dspammt(dummy,klr_intern.xdata(1:nn,:),xtemp);
      else
	fst_dspammt(dummy,klr_intern.xdata(ind,:),xtemp);
      end
      xtemp=dummy(:,:); % force copy
    end
  else
    if isempty(ind)
      xtemp=klr_intern.xdata*(klr_intern.xdata'*xtemp);
    elseif ind(1)==0
      xtemp=klr_intern.xdata(1:nn,:)* ...
	    (klr_intern.xdata(1:nn, :)'*xtemp);
    else
      xtemp=klr_intern.xdata(ind,:)* ...
	    (klr_intern.xdata(ind,:)'*xtemp);
    end
  end
  fst_muldiag(xtemp,vvec,0);
  if use_spars
    fst_addvec(xtemp,svec,0);
  end
elseif ~isfield(klr.hierarch,'lowrk') || ~klr.hierarch.lowrk.use
  fst_reshape(xtemp,nn*np,1);
  xtemp=klr_compkernmatvect(klr.num_data,xtemp,vvec,svec, ...
			    klr_intern.covmat,klr_intern.covtldiag, ...
			    klr.hierarch.l2p,ind,[], ...
			    klr_intern.covmul_tmpbuff);
  fst_reshape(xtemp,nn,np);
else
  fst_reshape(xtemp,nn*np,1);
  xtemp=klr_compkernmatvect_lr(klr.num_data,xtemp,vvec,svec, ...
			       klr_intern.covmat, ...
			       klr_intern.covtldiag, ...
			       klr.hierarch.lowrk.cmat_diag, ...
			       klr.hierarch.lowrk.perm, ...
			       klr.hierarch.lowrk.actsz, ...
			       klr.hierarch.l2p,ind,[], ...
			       klr_intern.covmul_tmpbuff);
  fst_reshape(xtemp,nn,np);
end
% Multiplication by PHI
klr_compmatphit(xtemp,klr.hierarch.childn);
y=reshape(xtemp(:,klr.hierarch.leafs),nn*nc,1);
