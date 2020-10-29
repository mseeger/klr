function crit = klr_accumcritgrad(targ,alpha,logpi,iind,jind, ...
				  oldcrit,varargin)
%KLR_ACCUMCRITGRAD Accumulate approx. CV criterion
%  CRIT = KLR_ACCUMCRITGRAD(TARG,ALPHA,LOGPI,IIND,JIND,OLDCRIT,
%         {CRITONLY=0})
%  The CV criterion is a sum of parts for subindexes I. Here, the
%  part for I is computed, added to OLDCRIT, and returned in
%  CRIT. I is given in IIND, its complement in JIND. TARG are all
%  (soft) targets, ALPHA the current primary parameters,
%  LOG(PI) given in LOGPI. Apart from the arguments, the setup is
%  given in the global structure KLR:
%  - KLR.COVMUL:     Function object for X -> K*X, K the kernel
%    matrix
%  - KLR.COVPARTMUL: Function object for X -> K(I,J)*X, where I, J
%    are subindexes
%  - KLR.NUM_DATA:   Number of datapoints
%  - KLR.NUM_CLASS:  Number of classes
%  - KLR.COVDIAG:    DIAG(K), K the kernel matrix
%  - KLR.VERBOSE:    Verbosity level. 0: No messages. 1: Some
%    messages. 2: Many messages
%  - KLR.LPITHRES:   Components in PI smaller than EXP(KLR.LPITHRES)
%    are considered to be 0 (numerical stability)
%  - KLR.CVCG_TOL:   Tolerance parameter for inner loop PCG
%  - KLR.CVCG_MAXIT: Max. number iterations in inner loop PCG
%  - KLR.BIAS_PVAR:  Variance param. for BIAS prior
%  The accumulator matrix for the gradient has form A*B' for
%  matrices A, B. They are kept in KLR_INTERN.ACCUM_AMAT and
%  KLR_INTERN.ACCUM_BMAT. The effect of an accumulation for I is to
%  append 3 columns to each A, B, this is done here as well. The
%  gradient accumulation is not done if CRITONLY==1.

global klr klr_intern;

% Initialization
if ~klr.covinfo.prec_ok
  error('Kernel precomp. matrix not up-to-date');
end
critonly=0;
if nargin>7
  critonly=varargin{1};
end
if size(jind,2)~=1
  jind=jind';
end
if size(iind,2)~=1
  iind=iind';
end
nc=klr.num_class; nn=klr.num_data; n=nn*nc;
nnj=length(jind); nj=nnj*nc;
temp=nn*(0:(nc-1));
jcind=reshape(jind(:,ones(nc,1))+temp(ones(nnj,1),:),nj,1);
nni=length(iind); ni=nni*nc;
icind=reshape(iind(:,ones(nc,1))+temp(ones(nni,1),:),ni,1);
solveexact=isfield(klr,'debug') && isfield(klr.debug,'solveexact') && ...
    klr.debug.solveexact;
if isfield(klr,'mixmat') && isfield(klr.mixmat,'use') && ...
      klr.mixmat.use
  usemix=1;
else
  usemix=0;
end
if ~solveexact
  if isfield(klr,'cvcg_tol')
    cgtol=klr.cvcg_tol;
  else
    cgtol=klr.cg_tol;
  end
  if isfield(klr,'cvcg_maxit')
    cgmaxit=klr.cvcg_maxit;
  else
    cgmaxit=klr.cg_maxit;
  end
end
% Precomputations for PCG solver (reduced system)
logpij=logpi(jcind); targj=targ(jcind);
alphaj=alpha(jcind);
pij=exp(logpij); sqpij=exp(0.5*logpij);
if ~solveexact
  % Special treatment for numerical stability
  temp=(logpij<klr.lpithres)&(targj>0);
  excind=find(temp); nexcind=find(1-temp);
  no_exc=isempty(excind);
  if klr.verbose>1 && ~no_exc
    fprintf(1,'PI too small on %d components, treated specially.\n',...
	    length(excind));
  end
  if ~no_exc
    rlogpi=logpij(nexcind); rtarg=targj(nexcind);
    ralpha=alphaj(nexcind);
    rpi=pij(nexcind); rsqpi=sqpij(nexcind);
  else
    rlogpi=logpij; rtarg=targj;
    ralpha=alphaj;
    rpi=pij; rsqpi=sqpij;
  end
  nsmallind=find(rlogpi>=klr.lpithres);
  % Diagonal of system matrix (for preconditioning)
  temp=zeros(nj,1); temp(nexcind)=exp(2*rlogpi);
  covdgj=klr.covdiag(jcind)+klr.bias_pvar;
  temp3=klr_mulp(temp.*covdgj,nnj,nc);
  klr_intern.diagmat=rpi.*((1-2*rpi).*covdgj(nexcind)+ ...
			   temp3(nexcind))+1;
  if klr.verbose>1
    fprintf(1,'Preconditioner: max=%f, min=%f\n', ...
	    max(klr_intern.diagmat),min(klr_intern.diagmat));
  end
  klr_intern.nexcind=nexcind;
  klr_intern.logpi=logpij; % Not reduced
  klr_intern.pi=pij;
  klr_intern.sqpi=sqpij;
  % Compute BETAJP (using PCG) and ALPHAJP
  % Compute init. BETAJP, r.h.s. RHS of system
  if length(nsmallind)==length(nexcind)
    betajp=ralpha.*exp(-0.5*rlogpi);
    temp=klr_tilcovpartmul(alpha(icind),jind,iind);
    klr_mexmulv(temp,1,nnj,nc,sqpij,pij);
    rhs=temp(nexcind);
  else
    betajp=zeros(length(nexcind),1);
    betajp(nsmallind)=ralpha(nsmallind).*exp(-0.5* ...
					     rlogpi(nsmallind));
    temp=klr_tilcovpartmul(alpha(icind),jind,iind);
    klr_mexmulv(temp,1,nnj,nc,sqpij,pij);
    rhs=zeros(length(nexcind),1);
    rrind=nexcind(nsmallind);
    rhs(nsmallind)=temp(rrind);
  end
  % Shuffling of kernel matrices s.t. JIND is on top
  feval(klr.covshuffle,jind,iind,1);
  % PCG run for BETAJP
  [betajp,flag,relres,nit]=pcg(@klr_mulsysmat,rhs,cgtol,...
			       cgmaxit,@klr_divprecond,[],betajp, ...
			       [0; nnj]);
  if klr.verbose>1
    if flag==0
      fprintf(1,'CG(BETAJP): Converged to req. tol.\n');
    elseif flag==1
      fprintf(1,'CG(BETAJP): Run for max. number of iter.\n');
    else
      fprintf(1,'CG(BETAJP): Terminated for other reason (flag=%d)\n',flag);
    end
    fprintf(1,'    Number of iter.: %d\n',nit);
    fprintf(1,'    Final rel. res.: %f\n',relres);
  end
  % De-shuffling
  feval(klr.covshuffle,jind,iind,0);
  % Compute new ALPHAJP from BETAJP
  alphajp=zeros(nj,1);
  alphajp(nexcind)=betajp;
  klr_mexmulv(alphajp,0,nnj,nc,sqpij,pij);
  alphajp=alphajp+alphaj;
  if ~no_exc
    temp=betajp; betajp=zeros(nj,1); betajp(nexcind)=temp;
  end
else
  % Compute ALPHAJP exactly (no BETAJP)
  klr_intern.pi=pij;
  rhs=klr_mulw(klr_tilcovpartmul(alpha(icind),jind,iind),nnj,nc);
  if ~usemix
    alphajp=alphaj+klr_solveredsystem(rhs,logpi,iind,jind);
  else
    alphajp=alphaj+klr_debug_solveredsystem(rhs,logpi,iind,jind);
  end
end
% Compute prediction on IIND and criterion CRIT
uip=klr_tilcovpartmul(alphajp,iind,jind);
lip=logsumexp(reshape(uip,nni,nc)')';
logpiip=uip-reshape(lip(:,ones(nc,1)),ni,1);
targi=targ(icind);
crit=oldcrit-targi'*uip+sum(lip);

if ~critonly
  temp=exp(logpiip)-targi;
  rvec=klr_tilcovpartmul(temp,jind,iind);
  qvec=alpha; qvec(jcind)=qvec(jcind)-alphajp;
  if ~solveexact
    % Compute S3VEC (using PCG) and S2VEC
    % Compute r.h.s. RHS of system
    temp=rvec(:,:);
    klr_mexmulv(temp,1,nnj,nc,sqpij,pij);
    rhs=temp(nexcind);
    % Shuffling of kernel matrices s.t. JIND is on top
    feval(klr.covshuffle,jind,iind,1);
    % PCG run (start from 0)
    svec=zeros(length(nexcind),1);
    [svec,flag,relres,nit]=pcg(@klr_mulsysmat,rhs,cgtol,...
			       cgmaxit,@klr_divprecond,[],svec, ...
			       [0; nnj]);
    if klr.verbose>1
      if flag==0
	fprintf(1,'CG(S3VEC): Converged to req. tol.\n');
      elseif flag==1
	fprintf(1,'CG(S3VEC): Run for max. number of iter.\n');
      else
	fprintf(1,'CG(S3VEC): Terminated for other reason (flag=%d)\n',flag);
      end
      fprintf(1,'    Number of iter.: %d\n',nit);
      fprintf(1,'    Final rel. res.: %f\n',relres);
    end
    s3vec=zeros(nj,1); s3vec(nexcind)=svec;
    % De-shuffling
    feval(klr.covshuffle,jind,iind,0);
    % Compute S2VEC
    klr_intern.pi=pij; % for MULM1, MULM2, MULV
    klr_intern.sqpi=sqpij;
    temp=klr_wrap_covmul(qvec);
    klr_addsigsq(temp,qvec,klr.bias_pvar,nn,nc);
    s2vec=klr_mulm2(s3vec,temp(jcind),nnj,nc);
    % We replace S3VEC with V_J*S3VEC here!
    klr_mexmulv(s3vec,0,nnj,nc,sqpij,pij);
    temp=rvec-klr_wrap_covmul(s3vec,jind);
    klr_addsigsq(temp,s3vec,-klr.bias_pvar,nnj,nc);
    s2vec=s2vec+klr_mulm1(temp,betajp,nnj,nc);
    klr_mexmulv(s2vec,1,nnj,nc,sqpij,pij);
    klr_mexmulv(s2vec,0,nnj,nc,sqpij,pij);
    % Precomputations for PCG solver (full system)
    pivec=exp(logpi); sqpi=exp(0.5*logpi);
    % Special treatment for numerical stability
    temp=(logpi<klr.lpithres)&(targ>0);
    excind=find(temp); nexcind=find(1-temp);
    no_exc=isempty(excind);
    if klr.verbose>1 && ~no_exc
      fprintf(1,'PI too small on %d components, treated specially.\n',...
	      length(excind));
    end
    if ~no_exc
      rlogpi=logpi(nexcind); rtarg=targ(nexcind);
      ralpha=alpha(nexcind);
      rpi=pivec(nexcind); rsqpi=sqpi(nexcind);
    else
      rlogpi=logpi; rtarg=targ;
      ralpha=alpha;
      rpi=pivec; rsqpi=sqpi;
    end
    % Diagonal of system matrix (for preconditioning)
    temp=zeros(n,1); temp(nexcind)=exp(2*rlogpi);
    covdg=klr.covdiag+klr.bias_pvar;
    temp3=klr_mulp(temp.*covdg,nn,nc);
    klr_intern.diagmat=rpi.*((1-2*rpi).*covdg(nexcind)+ ...
			     temp3(nexcind))+1;
    if klr.verbose>1
      fprintf(1,'Preconditioner: max=%f, min=%f\n', ...
	      max(klr_intern.diagmat),min(klr_intern.diagmat));
    end
    klr_intern.nexcind=nexcind;
    klr_intern.logpi=logpi; % Not reduced
    klr_intern.pi=pivec;
    klr_intern.sqpi=sqpi;
    % Compute S1VEC (using PCG on full system)
    % Compute RHS
    rhs=zeros(n,1);
    rhs(icind)=klr_tilcovpartmul(s3vec,iind,jind);
    temp=zeros(n,1); temp(jcind)=s2vec;
    rhs=rhs+klr_wrap_covmul(temp);
    klr_addsigsq(rhs,temp,klr.bias_pvar,nn,nc);
    rhs(jcind)=rhs(jcind)+rvec;
    klr_mexmulv(rhs,1,nn,nc,sqpi,pivec);
    rhs=rhs(nexcind);
    % PCG run (start from 0)
    svec=zeros(length(nexcind),1);
    [svec,flag,relres,nit]=pcg(@klr_mulsysmat,rhs,cgtol,...
			       cgmaxit,@klr_divprecond,[],svec);
    if klr.verbose>1
      if flag==0
	fprintf(1,'CG(S1VEC): Converged to req. tol.\n');
      elseif flag==1
	fprintf(1,'CG(S1VEC): Run for max. number of iter.\n');
      else
	fprintf(1,'CG(S1VEC): Terminated for other reason (flag=%d)\n',flag);
      end
      fprintf(1,'    Number of iter.: %d\n',nit);
      fprintf(1,'    Final rel. res.: %f\n',relres);
    end
    s1vec=zeros(n,1); s1vec(nexcind)=-svec;
    klr_mexmulv(s1vec,0,nn,nc,sqpi,pivec);
  else
    % Compute S vectors exactly
    klr_intern.pi=pij;
    if ~usemix
      s3vec=klr_solveredsystem(rvec,logpi,iind,jind,1,1);
    else
      s3vec=klr_debug_solveredsystem(rvec,logpi,iind,jind,1,1);
    end
    temp=klr_wrap_covmul(qvec);
    klr_addsigsq(temp,qvec,klr.bias_pvar,nn,nc);
    s2vec=klr_mulw(klr_mulm3(s3vec,temp(jcind),nnj,nc),nnj,nc);
    % Multiply S3VEC by W
    s3vec=klr_mulw(s3vec,nnj,nc);
    temp=zeros(n,1); temp(jcind)=s2vec;
    rhs=klr_wrap_covmul(temp);
    klr_addsigsq(rhs,temp,klr.bias_pvar,nn,nc);
    rhs(jcind)=rhs(jcind)+rvec;
    rhs(icind)=rhs(icind)+klr_tilcovpartmul(s3vec,iind,jind);
    klr_intern.pi=exp(logpi);
    if ~usemix
      s1vec=-klr_mulw(klr_solvesystem(rhs,logpi,1,1),nn,nc);
    else
      s1vec=-klr_mulw(klr_debug_solvesystem(rhs,logpi,1,1),nn,nc);
    end
  end
  % Accumulate AMAT and BMAT columns
  amat=zeros(n,3); bmat=zeros(n,3);
  amat(jcind,1)=alphajp;
  amat(:,2)=alpha;
  amat(:,3)=qvec;
  bmat(icind,1)=exp(logpiip)-targi;
  bmat(:,2)=s1vec;
  bmat(jcind,2)=s1vec(jcind)+s2vec;
  bmat(jcind,3)=s3vec;
  klr_intern.accum_amat=[klr_intern.accum_amat amat];
  klr_intern.accum_bmat=[klr_intern.accum_bmat bmat];
end
