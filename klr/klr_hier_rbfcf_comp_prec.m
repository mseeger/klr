function dummy = klr_hier_rbfcf_comp_prec(varargin)
%KLR_HIER_RBFCF_COMP_PREC COMP_PREC implementation for RBF kernel
%  KLR_HIER_RBFCF_COMP_PREC({NEWDATA=0})
%  Computes representation and diagonal for RBF kernel,
%  hierarchical classification. The common precomputation matrix
%  of all squared distances is stored in KLR_INTERN.PRECMAT.
%  The data matrix (cases as rows) must be given in
%  KLR_INTERN.XDATA.
%  NOTE: Here, all kernels must be RBF. A more general
%  implementation would allow kernels of different types.

global klr klr_intern;

if ~isfield(klr,'hierarch') || ~klr.hierarch.use
  error('Hierarchical classification mode not active');
end
if isfield(klr.hierarch,'lowrk') && klr.hierarch.lowrk
  error('Low rank mode not yet implemented');
end
newdata=0;
if nargin>0
  newdata=varargin{1};
end
nn=klr.num_data;
if nn~=size(klr_intern.xdata,1)
  error('KLR.NUM_DATA or KLR_INTERN.XDATA wrong');
end
if klr.verbose>1
  fprintf(1,'KLR_HIER_RBFCF_COMP_PREC: Doing precomputation.\n');
end

if newdata
  klr_intern.precmat=radialcf_precmat(klr_intern.xdata, ...
				      klr_intern.xdata);
else
  if size(klr_intern.precmat,1)~=nn || size(klr_intern.precmat,2)~= ...
	nn
    error('KLR_INTERN.PRECMAT has wrong size, need NEWDATA=1');
  end
end
nl=length(klr.hierarch.mpar_num);
l=1;
klr_intern.covmat=zeros(nn,nn*ceil(nl/2));
for i=1:floor(nl/2)
  wl=exp(klr.covinfo.theta(klr.hierarch.mpar_pos(l)));
  wu=exp(klr.covinfo.theta(klr.hierarch.mpar_pos(l+1)));
  ind=(nn*(i-1)+1):(nn*i);
  klr_intern.covmat(:,ind)=tril(exp(wl*klr_intern.precmat))+ ...
      triu(exp(wu*klr_intern.precmat),1);
  l=l+2;
end
if mod(nl,2)==1
  i=ceil(nl/2);
  wl=exp(klr.covinfo.theta(klr.hierarch.mpar_pos(l)));
  ind=(nn*(i-1)+1):(nn*i);
  klr_intern.covmat(:,ind)=exp(wl*klr_intern.precmat);
end
klr_intern.covtldiag=ones(nn,nl);
klr_hier_compcovdiag;
klr.covinfo.prec_ok=1;
