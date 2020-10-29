function y = klr_rbfcf_covmmulmat(x,l,tedata,varargin)
%KLR_RBFCF_COVMMULMAT COVMMULMAT implementation for RBF kernel
%  Y = KLR_RBFCF_COVMMULMAT(X,L,TEDATA,{IND})
%  Computes Y = M*X, M the M^(l) matrix between test points
%  TEDATA and training points in KLR_INTERN.XDATA. Here, l==L.
%  If IND is given, it is a subindex applied to the training
%  points

global klr klr_intern;

if ~isfield(klr,'hierarch') || ~klr.hierarch.use
  error('Hierarchical classification mode not active');
end
nn=klr.num_data;
if size(tedata,2)~=size(klr_intern.xdata,2)
  error('TEDATA has wrong size');
end
if l<1 || l>length(klr.hierarch.mpar_num)
  error('L wrong');
end
ind=[]; nx=nn;
if nargin>3
  ind=varargin{1};
  nx=length(ind);
end
if size(x,1)~=nx
  error('X has wrong size');
end

if isempty(ind)
  y=radialcf(tedata,klr_intern.xdata,1, ...
	     exp(klr.covinfo.theta(klr.hierarch.mpar_pos(l))))*x;
else
  y=radialcf(tedata,klr_intern.xdata(ind,:),1, ...
	     exp(klr.covinfo.theta(klr.hierarch.mpar_pos(l))))*x;
end
