function y = klr_debug_solveredsystem(x,logpi,iind,jind,varargin)
%KLR_DEBUG_SOLVEREDSYSTEM Solve linear system exactly
%  See KLR_SOLVEREDSYSTEM, but for case of mixing matrix.

global klr klr_intern;

uselast=0; trs=0;
if ~isfield(klr,'mixmat') || ~isfield(klr.mixmat,'use') || ~ ...
      klr.mixmat.use
  error('No mixing matrix');
end
nn=length(jind);
nc=klr.num_class; n=nn*nc;
if nargin>4
  uselast=varargin{1};
  if nargin>5
    trs=varargin{2};
  end
end
if klr.mixmat.use==2
  i=klr.mixmat.thpos;
  klr.mixmat.bmat=reshape(klr.covinfo.theta(i:(i+nc*nc-1)),nc,nc);
end
bmat=klr.mixmat.bmat;

tvec=(0:(nc-1))*klr.num_data;
jcind=reshape(jind(:,ones(nc,1))+tvec(ones(nn,1),:),n,1);
pivec=exp(logpi(jcind));
hmat=zeros(n,n);
for c=1:nc
  rng=((c-1)*nn+1):(c*nn);
  hmat(rng,rng)=feval(klr.getcovmat,c,jind);
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
  tmat=reshape(sum(reshape(hmat,n*nn,nc),2),n,nn);
  for c=1:nc
    rng=((c-1)*nn+1):(c*nn);
    hmat(:,rng)=hmat(:,rng)-muldiag(tmat,pivec(rng));
  end
end
dgind=(0:(n-1))'*(n+1)+1;
hmat(dgind)=hmat(dgind)+1;
y=hmat\x;
