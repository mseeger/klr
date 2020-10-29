function y = klr_gen_covmul(x,varargin)
%KLR_GEN_COVMUL Generic COVMUL implementation
%  Y = KLR_GEN_COVMUL(X,{IND})
%  Computes Y = K*X for the kernel matrix K. K is block-diagonal,
%  the blocks are obtained from KLR_INTERN.COVMAT. If
%  KLR.COVINFO.TIED==1, KLR_INTERN.COVMAT contains the single
%  block. Otherwise, KLR_INTERN.COVMAT is a cell array containing
%  the successive blocks in lower and upper triangles (odd ones in
%  lower). Diagonals in KLR.COVDIAG.
%  If IND is given and not empty, it is an index selecting
%  submatrices of the kernel matrix blocks. X, Y are smaller in
%  this case.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

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
    else
      [nn,b]=size(ind);
      if b~=1
	ind=ind';
	nn=b;
      end
    end
  end
end
nc=klr.num_class; n=nn*nc;
if length(x)~=n
  error('X has wrong size');
end
y=zeros(n,1);
if ~klr.covinfo.tied
  if isempty(ind)
    dgind=(0:(nn-1))'*(nn+1)+1;
    actc=1; lower=1;
    for c=1:nc
      rng=(nn*(c-1)+1):(nn*c);
      klr_intern.covmat{actc}(dgind)=klr.covdiag(rng);
      y(rng)=klr_compmatvect(klr_intern.covmat{actc},x(rng), ...
			     double(lower));
      if lower
	lower=0;
      else
	lower=1;
	actc=actc+1;
      end
    end
  elseif ind(1)==0
    dgind=(0:(nn-1))'*(klr.num_data+1)+1;
    lower=1; actc=0;
    for c=1:nc
      if lower
	actc=actc+1;
      end
      rng=(nn*(c-1)+1):(nn*c);
      rng2=(klr.num_data*(c-1)+1):(klr.num_data*(c-1)+nn);
      klr_intern.covmat{actc}(dgind)=klr.covdiag(rng2);
      y(rng)=klr_compmatvect(klr_intern.covmat{actc},x(rng), ...
			     double(lower),ind);
      lower=~lower;
    end
  else
    dgind=(0:(nn-1))'*(nn+1)+1;
    lower=1; actc=0;
    for c=1:nc
      if lower
	actc=actc+1;
	% Slow:
	tmat=klr_intern.covmat{actc}(ind,ind);
      end
      rng=(nn*(c-1)+1):(nn*c);
      tmat(dgind)=klr.covdiag(ind+(c-1)*nn);
      y(rng)=klr_compmatvect(tmat,x(rng),double(lower));
      lower=~lower;
    end
  end
else
  fst_reshape(x,nn,nc);
  fst_reshape(y,nn,nc);
  if isempty(ind) || ind(1)==0
    fst_dsymm(y,{klr_intern.covmat; [1; 1; nn; nn]; 'L '},x);
  else
    fst_dsymm(y,{klr_intern.covmat(ind,ind); [1; 1; nn; nn]; 'L '}, ...
	      x);
  end
  fst_reshape(x,n,1);
  fst_reshape(y,n,1);
end
