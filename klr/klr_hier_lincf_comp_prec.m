function dummy = klr_hier_lincf_comp_prec(varargin)
%KLR_HIER_LINCF_COMP_PREC COMP_PREC implementation for linear kernel
%  KLR_HIER_LINCF_COMP_PREC({NEWDATA=0})
%  Computes representation and diagonal for linear kernel,
%  hierarchical classification. The linear kernel is just the usual
%  Euclidean inner product.
%  The data matrix (cases as rows) must be given in
%  KLR_INTERN.XDATA.

global klr klr_intern;

if ~isfield(klr,'hierarch') || ~klr.hierarch.use
  error('Hierarchical classification mode not active');
end
if ~isfield(klr.hierarch,'linear_kern') || ~klr.hierarch.linear_kern
  error('Linear kernel mode not active');
end
newdata=0;
if nargin>0
  newdata=varargin{1};
end
if newdata
  klr_intern.covtldiag=sum(klr_intern.xdata.*klr_intern.xdata,2);
end
klr_hier_compcovdiag;
klr.covinfo.prec_ok=1;
