function dummy = klr_hier_mlpcf_comp_prec(varargin)
%KLR_HIER_MLPCF_COMP_PREC COMP_PREC implementation for MLP kernel
%  KLR_HIER_MLPCF_COMP_PREC({NEWDATA=0})
%  Computes representation and diagonal for MLP kernel,
%  hierarchical classification.
%  The data matrix (cases as rows) must be given in
%  KLR_INTERN.XDATA.

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
  fprintf(1,'KLR_HIER_MLPCF_COMP_PREC: Doing precomputation.\n');
end

precmat=klr_intern.xdata*klr_intern.xdata';
dgx=diag(precmat);
nl=length(klr.hierarch.mpar_num);
l=1;
klr_intern.covmat=zeros(nn,nn*ceil(nl/2));
klr_intern.covtldiag-zeros(nn,nl);
for i=1:floor(nl/2)
  wl=exp(klr.covinfo.theta(klr.hierarch.mpar_pos(l)));
  wu=exp(klr.covinfo.theta(klr.hierarch.mpar_pos(l+1)));
  ind=(nn*(i-1)+1):(nn*i);
  dummy=precmat+wl;
  dvec=1./sqrt(dgx+(wl+0.5));
  fst_muldiag(dummy,dvec,0);
  fst_muldiag(dummy,dvec,1);
  klr_intern.covmat(:,ind)=tril(asin(dummy));
  klr_intern.covtldiag(:,l)=diag(klr_intern.covmat(:,ind));
  dummy=precmat+wu;
  dvec=1./sqrt(dgx+(wu+0.5));
  fst_muldiag(dummy,dvec,0);
  fst_muldiag(dummy,dvec,1);
  klr_intern.covmat(:,ind)=klr_intern.covmat(:,ind)+ ...
      triu(asin(dummy));
  klr_intern.covtldiag(:,l+1)=diag(klr_intern.covmat(:,ind));
  l=l+2;
end
if mod(nl,2)==1
  i=ceil(nl/2);
  wl=exp(klr.covinfo.theta(klr.hierarch.mpar_pos(nl)));
  ind=(nn*(i-1)+1):(nn*i);
  dummy=precmat+wl;
  dvec=1./sqrt(dgx+(wl+0.5));
  fst_muldiag(dummy,dvec,0);
  fst_muldiag(dummy,dvec,1);
  klr_intern.covmat(:,ind)=tril(asin(dummy));
  klr_intern.covtldiag(:,nl)=diag(klr_intern.covmat(:,ind));
end
klr_hier_compcovdiag;
klr.covinfo.prec_ok=1;
