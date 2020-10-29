function y = klr_solveredsystem(x,logpi,iind,jind,varargin)
%KLR_SOLVEREDSYSTEM Solve reduced linear system exactly
%  Y = KLR_SOLVEREDSYSTEM(X,LOGPI,IIND,JIND,{USELAST=0},{TRS=0})
%  Does the same as KLR_SOLVESYSTEM, but reduced to the components in
%  JIND. IIND is the complement of JIND. Full LOG(PI) given in
%  LOGPI. Requires inv. matrices for full system in KLR_INTERN.INVMAT. If
%  USELAST==0, some precomp. is stored in KLR_INTERN.REDFACTQ,
%  KLR_INTERN.REDFACTR. If USELAST==1, these are read from
%  there. If TRS==1, the system matrix is transposed.

global klr klr_intern;

uselast=0; trs=0;
nn=klr.num_data; nc=klr.num_class; n=nn*nc;
if nargin>4
  uselast=varargin{1};
  if nargin>5
    trs=varargin{2};
  end
end
if ~isfield(klr_intern,'invmat') || length(klr_intern.invmat)~=nc+1
  error('Need KLR_INTERN.INVMAT');
end
if uselast && (~isfield(klr_intern,'redfactq') || ~ ...
	      isfield(klr_intern,'redfactr'))
  error('Need KLR_INTERN.REDFACT variables with USELAST==1');
end
%if isfield(klr,'mixmat') & isfield(klr.mixmat,'use') & ...
%      klr.mixmat.use
%  error('Cannot do exact solution with mixing matrix');
%end
sqpi=exp(0.5*logpi);
nnj=length(jind); nj=nnj*nc;
nni=length(iind); ni=nni*nc;

% Precomputation (QR decomp. of (H^-1)_I)
if ~uselast
  klr_intern.redfactq=zeros(ni,ni);
  v2mat=zeros(nn,ni); v1mat=zeros(nni,ni);
  for c=1:nc
    rng=((c-1)*nn+1):(nn*c);
    rngi=((c-1)*nni+1):(nni*c);
    icind=iind+(c-1)*nn;
    v2mat(:,rngi)=muldiag(sqpi(rng),muldiag(klr_intern.invmat{c}(:, ...
						  iind),exp(-0.5* ...
						  logpi(icind))));
    v1mat(:,rngi)=v2mat(iind,rngi);
    tind=nn*(0:(nni-1))'+iind+(c-1)*nn*nni;
    v2mat(tind)=v2mat(tind)-1;
  end
  v2mat=-klr_intern.invmat{nc+1}\(klr_intern.invmat{nc+1}'\v2mat);
  for c=1:nc
    rng=((c-1)*nn+1):(nn*c);
    rngi=((c-1)*nni+1):(nni*c);
    icind=iind+(c-1)*nn;
    klr_intern.redfactq(rngi,:)= ...
	muldiag(sqpi(icind),klr_intern.invmat{c}(iind,:)* ...
		muldiag(sqpi(rng),v2mat));
    klr_intern.redfactq(rngi,rngi)=klr_intern.redfactq(rngi,rngi)+ ...
	v1mat(:,rngi);
  end
  [klr_intern.redfactq,klr_intern.redfactr]= ...
      qr(klr_intern.redfactq); % QR decomp.
end

% Solve system
if ~trs
  v1vec=zeros(n,1); v2vec=zeros(nn,1);
  for c=1:nc
    rng=((c-1)*nn+1):(nn*c);
    rngj=((c-1)*nnj+1):(nnj*c);
    jcind=jind+(c-1)*nn;
    xc=x(rngj);
    v1vec(rng)=sqpi(rng).*(klr_intern.invmat{c}(:,jind)*(exp(-0.5* ...
						  logpi(jcind)).* ...
						  xc));
    v2vec=v2vec-v1vec(rng);
    v2vec(jind)=v2vec(jind)+xc;
  end
  v2vec=klr_intern.invmat{nc+1}\(klr_intern.invmat{nc+1}'\v2vec);
  v3vec=zeros(n,1);
  for c=1:nc
    rng=((c-1)*nn+1):(nn*c);
    sqc=sqpi(rng);
    v3vec(rng)=sqc.*(klr_intern.invmat{c}*(sqc.*v2vec));
  end
  tmat=reshape(v3vec+v1vec,nn,nc);
  y=reshape(tmat(jind,:),nj,1); % first part
  x2=klr_intern.redfactr\(klr_intern.redfactq'*reshape(tmat(iind,:), ...
						  ni,1));
  v1vec=zeros(n,1); v2vec=zeros(nn,1);
  for c=1:nc
    rng=((c-1)*nn+1):(nn*c);
    rngi=((c-1)*nni+1):(nni*c);
    icind=iind+(c-1)*nn;
    xc=x2(rngi);
    v1vec(rng)=sqpi(rng).*(klr_intern.invmat{c}(:,iind)*(exp(-0.5* ...
						  logpi(icind)).* ...
						  xc));
    v2vec=v2vec-v1vec(rng);
    v2vec(iind)=v2vec(iind)+xc;
  end
  v2vec=klr_intern.invmat{nc+1}\(klr_intern.invmat{nc+1}'\v2vec);
  v3vec=zeros(nj,1);
  for c=1:nc
    rng=((c-1)*nn+1):(nn*c);
    rngj=((c-1)*nnj+1):(nnj*c);
    jcind=jind+nn*(c-1);
    v3vec(rngj)=sqpi(jcind).*(klr_intern.invmat{c}(jind,:)* ...
			      (sqpi(rng).*v2vec))+v1vec(jcind);
  end
  y=y-v3vec;
else
  % System with transposed matrix
  v1vec=zeros(n,1); v2vec=zeros(nn,1);
  for c=1:nc
    rng=((c-1)*nn+1):(nn*c);
    rngj=((c-1)*nnj+1):(nnj*c);
    jcind=jind+(c-1)*nn;
    v1vec(rng)=klr_intern.invmat{c}(:,jind)*(sqpi(jcind).*x(rngj));
    v2vec=v2vec+sqpi(rng).*v1vec(rng);
  end
  v2vec=klr_intern.invmat{nc+1}\(klr_intern.invmat{nc+1}'\v2vec);
  v3vec=zeros(n,1);
  for c=1:nc
    rng=((c-1)*nn+1):(nn*c);
    tvec=sqpi(rng).*v2vec;
    v3vec(rng)=exp(-0.5*logpi(rng)).*(v1vec(rng)+tvec- ...
				      klr_intern.invmat{c}*tvec);
  end
  tmat=reshape(v3vec,nn,nc);
  y=reshape(tmat(jind,:),nj,1); % first part
  x2=klr_intern.redfactq*(klr_intern.redfactr'\reshape(tmat(iind,:), ...
						  ni,1));
  v1vec=zeros(n,1); v2vec=zeros(nn,1);
  for c=1:nc
    rng=((c-1)*nn+1):(nn*c);
    rngi=((c-1)*nni+1):(nni*c);
    icind=iind+(c-1)*nn;
    v1vec(rng)=klr_intern.invmat{c}(:,iind)*(sqpi(icind).* ...
					     x2(rngi));
    v2vec=v2vec+sqpi(rng).*v1vec(rng);
  end
  v2vec=klr_intern.invmat{nc+1}\(klr_intern.invmat{nc+1}'\v2vec);
  v3vec=zeros(nj,1);
  for c=1:nc
    rng=((c-1)*nn+1):(nn*c);
    rngj=((c-1)*nnj+1):(nnj*c);
    jcind=jind+nn*(c-1);
    tvec=sqpi(rng).*v2vec;
    v3vec(rngj)=exp(-0.5*logpi(jcind)).*(v1vec(jcind)+tvec(jind)- ...
					 klr_intern.invmat{c}(jind,:)*tvec);
  end
  y=y-v3vec;
end
