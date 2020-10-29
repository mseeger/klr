function y = klr_rbfcf_derivmmulmat(x,l,ppos)
%KLR_RBFCF_DERIVMMULMAT DERIVMMULMAT implementation for RBF kernel
%  Y = KLR_RBFCF_DERIVMMULMAT(X,L,PPOS)
%  Computes Y = DV*X, DV the derivative of M^(l), l==L, w.r.t. the
%  PPOS-th parameter of this kernel.
%  The RBF kernel has a single parameter log(w), w the inv. squared
%  length scale. The precomp. matrix (see RADIALCF_PRECMAT) for the
%  data must be given in KLR_INTERN.PRECMAT.

global klr klr_intern;

if ~isfield(klr,'hierarch') || ~klr.hierarch.use
  error('Hierarchical classification mode not active');
end
nn=klr.num_data;
if l<1 || l>length(klr.hierarch.mpar_num)
  error('L wrong');
end
if ppos~=1
  error('PPOS wrong: RBF kernel has single parameter only');
end
if size(x,1)~=nn
  error('X has wrong size');
end
nq=size(x,2);

i=ceil(l/2);
if mod(l,2)==1
  uplo='L ';
else
  uplo='U ';
end
% Diag. of PRECMAT is 0
dummy=klr_intern.precmat.*klr_intern.covmat(:,(nn*(i-1)+1):(nn*i));
w=exp(klr.covinfo.theta(klr.hierarch.mpar_pos(l)));
y=zeros(nn,nq);
fst_dsymm(y,{dummy; [1; 1; nn; nn]; uplo},x,w);
