function y = klr_solvesystem(x,logpi,varargin)
%KLR_SOLVESYSTEM Solve linear system exactly
%  Y = KLR_SOLVESYSTEM(X,LOGPI,{USELAST=0},{TRS=0})
%  Returns Y = B\X for the system matrix B = I + W*K, K the sum of
%  kernel matrix and KLR.BIAS_PVAR times P_data. LOG(PI) given in LOGPI.
%  Requires computation of C inverses and 1 Chol. factor, stored in
%  KLR_INTERN.INVMAT, C the number of classes. If USELAST==1, the
%  factors are read from there. If TRS==1, the system matrix is
%  B = I + K*W instead.

global klr klr_intern;

uselast=0; trs=0;
subind=[];
if isfield(klr,'worksubind') && ~isempty(klr.worksubind)
  subind=klr.worksubind;
  nn=length(subind);
else
  nn=klr.num_data;
end
nc=klr.num_class; n=nn*nc;
if nargin>2
  uselast=varargin{1};
  if nargin>3
    trs=varargin{2};
  end
end
if uselast && (~isfield(klr_intern,'invmat') || length(klr_intern.invmat)~= ...
	      nc+1)
  error('Need KLR_INTERN.INVMAT for USELAST==1');
end
%if isfield(klr,'mixmat') & isfield(klr.mixmat,'use') & ...
%      klr.mixmat.use
%  error('Cannot do exact solution with mixing matrix');
%end
sqpi=exp(0.5*logpi);
if ~uselast
  if isfield(klr_intern,'invmat') && ...
	length(klr_intern.invmat)~=nc+1
    klr_intern.invmat=[];
  end
  klr_intern.invmat{nc+1}=zeros(nn);
  dgind=(0:(nn-1))'*(nn+1)+1;
  for c=1:nc
    rng=((c-1)*nn+1):(c*nn);
    dsqvec=sqpi(rng);
    tmat=chol(muldiag(dsqvec,muldiag(feval(klr.getcovmat,c, ...
					   subind)+klr.bias_pvar, ...
				     dsqvec))+eye(nn));
    if fst_invchol({tmat,[1 1 nn nn],'UN'})~=0
      error('Error computing INVCHOL');
    end
    klr_intern.invmat{c}=makesymm(tmat,0);
    klr_intern.invmat{nc+1}=klr_intern.invmat{nc+1}+ ...
	muldiag(dsqvec,muldiag(klr_intern.invmat{c},dsqvec));
  end
  klr_intern.invmat{nc+1}=chol(klr_intern.invmat{nc+1});
end
y=zeros(n,1);
if ~trs
  % B = I + W*K
  tvec=x./sqpi;
  for c=1:nc
    ind=((c-1)*nn+1):(c*nn);
    y(ind)=klr_intern.invmat{c}*tvec(ind);
  end
  tvec=sqpi.*(tvec-y);
  tvec2=klr_intern.invmat{nc+1}\(klr_intern.invmat{nc+1}'\ ...
				 sum(reshape(tvec,nn,nc),2));
  for c=1:nc
    ind=((c-1)*nn+1):(c*nn);
    dsqvec=sqpi(ind);
    y(ind)=dsqvec.*(y(ind)+klr_intern.invmat{c}*(dsqvec.*tvec2));
  end
else
  % B = I + K*W
  tvec=x.*sqpi;
  for c=1:nc
    ind=((c-1)*nn+1):(c*nn);
    y(ind)=klr_intern.invmat{c}*tvec(ind);
  end
  tvec=sqpi.*y;
  tvec2=klr_intern.invmat{nc+1}\(klr_intern.invmat{nc+1}'\ ...
				 sum(reshape(tvec,nn,nc),2));
  for c=1:nc
    ind=((c-1)*nn+1):(c*nn);
    dsqvec=sqpi(ind);
    tvec=dsqvec.*tvec2;
    y(ind)=(y(ind)+tvec-klr_intern.invmat{c}*tvec)./dsqvec;
  end
end
