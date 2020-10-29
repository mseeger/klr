function y = klr_covfunlr(x,varargin)
%KLR_COVFUNLR Internal helper for KLR_FINDMAP
%  Y = KLR_COVFUNLR(X,{IND})
%  Computes Y = K*X for the kernel matrix K. K is block-diagonal,
%  each block is a low-rank matrix of the form U*U'. The U matrices
%  are entries of the cell array KLR_INTERN.COVFACT. If this has a
%  single entry only, all blocks are the same.
%  If IND is given and not empty, it is an index selecting
%  submatrices of the kernel matrix blocks (rows of the U
%  factors). X, Y are smaller in this case.

global klr klr_intern;

nn=klr.num_data;
ind=[];
nn=klr.num_data;
if nargin>1
  ind=varargin{1};
  if ~isempty(ind)
    [nn,b]=size(ind);
    if b~=1
      ind=ind'; nn=b;
    end
  end
end
nc=klr.num_class; n=nn*nc;
y=zeros(n,1);
if isempty(ind)
  if length(klr_intern.covfact)~=1
    for c=1:nc
      rng=(nn*(c-1)+1):(nn*c);
      y(rng)=klr_intern.covfact{c}*(klr_intern.covfact{c}'*x(rng));
    end
  else
    y=reshape(klr_intern.covfact{1}*(klr_intern.covfact{1}'* ...
				     reshape(x,nn,nc)),n,1);
  end
else
  if length(klr_intern.covfact)~=1
    for c=1:nc
      rng=(nn*(c-1)+1):(nn*c);
      temp=klr_intern.covfact{c}(ind,:);
      y(rng)=temp*(temp'*x(rng));
    end
  else
    temp=klr_intern.covfact{1}(ind,:);
    y=reshape(temp*(temp'*reshape(x,nn,nc)),n,1);
  end
end
