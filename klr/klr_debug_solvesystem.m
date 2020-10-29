function y = klr_debug_solvesystem(x,logpi,varargin)
%KLR_SOLVESYSTEM Solve linear system exactly
%  Y = KLR_SOLVESYSTEM(X,LOGPI,{USELAST=0},{TRS=0})
%  Debug version of KLR_SOLVESYSTEM, works also with mixing matrix.
%  B is constructed and inverted explicitly. Only for small sets!

global klr klr_intern;

uselast=0; trs=0;
if isfield(klr,'worksubind') && ~isempty(klr.worksubind)
  error('Not supported!');
end
if ~isfield(klr,'mixmat') || ~isfield(klr.mixmat,'use') || ~ ...
      klr.mixmat.use
  error('No mixing matrix');
end
nc=klr.num_class; nn=klr.num_data; n=nn*nc;
if nargin>2
  uselast=varargin{1};
  if nargin>3
    trs=varargin{2};
  end
end
if klr.mixmat.use==2
  i=klr.mixmat.thpos;
  klr.mixmat.bmat=reshape(klr.covinfo.theta(i:(i+nc*nc-1)),nc,nc);
end
bmat=klr.mixmat.bmat;

pivec=exp(logpi);
hmat=zeros(n,n);
for c=1:nc
  rng=((c-1)*nn+1):(c*nn);
  hmat(rng,rng)=feval(klr.getcovmat,c);
end
hmat=reshape(permute(reshape(reshape(permute(reshape(hmat,nn,nc,n),[1 ...
		    3 2]),nn*n,nc)*bmat',nn,n,nc),[1 3 2]),n,n);
hmat=reshape(reshape(hmat,nn*n,nc)*bmat',n,n);
ovec=ones(nn,1);
for c=1:nc
  rng=((c-1)*nn+1):(c*nn);
  hmat(rng,rng)=hmat(rng,rng)+klr.bias_pvar(ovec,ovec);
end
if ~trs
  hmat=muldiag(pivec,hmat);
  tmat=reshape(sum(reshape(hmat,nn,nc,n),2),nn,n);
  for c=1:nc
    rng=((c-1)*nn+1):(c*nn);
    hmat(rng,:)=hmat(rng,:)-muldiag(pivec(rng),tmat);
  end
else
  hmat=muldiag(hmat,pivec);
  tmat=reshape(sum(reshape(hmat,n,nn,nc),3),n,nn);
  for c=1:nc
    rng=((c-1)*nn+1):(c*nn);
    hmat(:,rng)=hmat(:,rng)-muldiag(tmat,pivec(rng));
  end
end
dgind=(0:(n-1))'*(n+1)+1;
hmat(dgind)=hmat(dgind)+1;
y=hmat\x;
