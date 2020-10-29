function [alpha,varargout] = klr_findmap(targ,salpha)
%KLR_FINDMAP Newton-Raphson maximization of log posterior
%  [ALPHA,{FLAG},{LOGPI},{UVEC}] = KLR_FINDMAP(TARG,SALPHA)
%  Runs Newton-Raphson to find posterior mode for multiple kernel
%  logistic regression model with soft targets TARG. Apart from the
%  arguments, the setup is given in the global structure KLR:
%  - KLR.COVMUL:    Function object for X -> K*X, K the kernel
%    matrix
%  - KLR.NUM_DATA:  Number of datapoints
%  - KLR.NUM_CLASS: Number of classes
%  - KLR.COVDIAG:   DIAG(K), K the kernel matrix
%  Vectors (ALPHA, TARG, UVEC, ...) are of size KLR.NUM_DATA *
%  KLR.NUM_CLASS. The ordering is s.t. the inner index is over
%  datapoints, the outer index is over classes (the kernel matrix K
%  is block-diagonal in that ordering).
%  The optimization is run in the dual variables ALPHA, where
%  F = K*ALPHA. We start with ALPHA==SALPHA. The intercept parameters
%  BIAS are a function of ALPHA. The prior for BIAS is N(0,s^2 I),
%  s^2==KLR.BIAS_PVAR.
%
%  Function returns the final ALPHA. In FLAG, we return 0 if NR
%  optimization attains tolerance KLR.TOL, 1 if it runs for
%  KLR.MAXITER iterations, or 2 if a significant criterion increase
%  or a complete screw-up (NaN) is detected. Note that a single
%  upwards jump in the criterion value is tolerated if followed by
%  a decrease. If FLAG==2, the other return values are
%  undefined. In LOGPI, UVEC, the final values of LOG(PI), U are
%  returned.
%  If KLR.RETRACT is given and ~=0, it must be a value in (0,1).
%  In this case, if the current step to ALPHA leads to a crit.
%  increase or screw-up from the prev. OALPHA, we try again with
%  l*OALPHA + (1-l)*ALPHA, l==KLR.RETRACT. This is done at most 3
%  times in a row. Note that a retract is very fast, because no
%  new system has to be solved.
%
%  Other fields required in KLR:
%  - KLR.TOL:       NR (outer optimization) terminates if relative
%    improvement in criterion value is below TOL
%  - KLR.MAXITER:   NR (outer optimization) terminates if MAXITER
%    iterations have been done
%  - KLR.BIAS_PVAR: Variance parameter for BIAS prior
%  - KLR.VERBOSE:   Verbosity level. 0: No messages. 1: Crit. value
%    at each iter. 2: Messages after each CG run
%  - KLR.LPITHRES:  Components in PI smaller than EXP(KLR.LPITHRES)
%    are considered to be 0 (numerical stability)
%  - KLR.CG_TOL:    Tolerance parameter for inner loop PCG
%  - KLR.CG_MAXIT:  Max. number iterations in inner loop PCG
%
%  Working on subset:
%  If KLR.WORKSUBIND is given and not empty, this is taken as index
%  into the training set. Everything is done w.r.t. this index
%  then, and variables ALPHA, LOGPI, UVEC, etc. are relative to
%  this index, also TARG must be.

global klr klr_intern;

% Initialization. Sanity checks
if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
if isfield(klr,'mixmat') && isfield(klr.mixmat,'use') && ...
      klr.mixmat.use
  usemix=1;
else
  usemix=0;
end
useworksub=0;
if isfield(klr,'worksubind') && ~isempty(klr.worksubind)
  useworksub=1;
  subind=klr.worksubind;
  nn=length(subind);
else
  nn=klr.num_data;
end
nc=klr.num_class; n=nn*nc;
[d1,d2]=size(salpha);
if d1~=n || d2~=1
  error('SALPHA has wrong size');
end
[d1,d2]=size(targ);
if d1~=n || d2~=1
  error('TARG has wrong size');
end
if ~isempty(find((sum(reshape(targ,nn,nc),2)-ones(nn,1))>1e-15))
  error('TARG entries must be normalized distributions');
end
if klr.tol<=0 || klr.maxiter<1
  error('KLR.TOL, KLR.MAXITER wrong');
end
if klr.bias_pvar<=0 || klr.verbose<0 || klr.lpithres>=0
  error('KLR.BIAS_PVAR, KLR.VERBOSE, KLR.LPITHRES wrong');
end
if klr.cg_tol<=0 || klr.cg_maxit<1
  error('KLR.CG_TOL, KLR.CG_MAXIT wrong');
end
% Project ALPHA onto range(W), just to make sure
alpha=salpha;
temp=mean(reshape(alpha,nn,nc),2);
alpha=alpha-reshape(temp(:,ones(nc,1)),n,1);

% Main loop (NR iterations)
solveexact=isfield(klr,'debug') && isfield(klr.debug,'solveexact') && ...
    klr.debug.solveexact;
do_retract=isfield(klr,'retract') && klr.retract>0;
iter=0;
crit=0;
convflag=0;
uvec=klr_wrap_covmul(alpha);
klr_addsigsq(uvec,alpha,klr.bias_pvar,nn,nc);
num_retract=0; % How many retract in a row?
while 1
  % Compute LOG(PI)
  l=logsumexp(reshape(uvec,nn,nc)')';
  logpi=uvec-reshape(l(:,ones(nc,1)),n,1);
  pivec=exp(logpi); sqpi=exp(0.5*logpi);
  % New criterion value. Stopping crit.
  oldcrit=crit;
  crit=uvec'*(0.5*alpha-targ)+sum(l);
  if klr.verbose>0
    fprintf(1,'Iteration %d: crit=%f\n',iter,crit);
  end
  % Test for convergence
  if iter>0 && klr.verbose>0
    % DEBUG
    fprintf(1,'FINDMAP: acc=%f\n',abs((crit-oldcrit)/oldcrit));
  end
  if iter>0 && abs(crit-oldcrit)<=klr.tol*abs(oldcrit)
    break; % Rel. improvement small enough
  elseif iter>=klr.maxiter
    convflag=1;
    break; % Max. number of iterations done
  end
  if do_retract && iter>0
    if num_retract<=2 && (isinf(crit) || isnan(crit) || crit> ...
			  oldcrit)
      % Try again with ALPHA closer to OALPHA
      alpha=(1-klr.retract)*alpha+klr.retract*oalpha;
      uvec=klr_wrap_covmul(alpha);
      klr_addsigsq(uvec,alpha,klr.bias_pvar,nn,nc);
      num_retract=num_retract+1;
      crit=oldcrit; % Value at OALPHA
      if klr.verbose>0
	fprintf(1,'FINDMAP: Crit. increase -> retract.\n');
      end
      continue;
    end
    if ~isnan(crit) && ~isinf(crit) && crit<=oldcrit-0.05* ...
	  abs(oldcrit)
      % Progress: reset functionality
      num_retract=0;
    end
  end
  if isinf(crit) || isnan(crit)
    % Serious screw-up. Note that this is after retract, or in the
    % first iter.
    convflag=2;
    break;
  elseif iter==0 || crit<mincrit
    mincrit=crit;
    numbad=0;
  elseif crit>mincrit+0.1*abs(mincrit)
    numbad=numbad+1;
    if numbad>2 || (numbad>1 && crit>mincrit+abs(mincrit))
      convflag=2;
      break; % Significant criterion increase
    end
  end
  % NR step for ALPHA
  if ~solveexact
    % Special treatment for numerical stability
    temp=(logpi<klr.lpithres)&(targ>0);
    excind=find(temp); nexcind=find(1-temp);
    no_exc=isempty(excind);
    if klr.verbose>0 && ~no_exc
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
    nsmallind=find(rlogpi>=klr.lpithres);
    % Compute init. BETA, r.h.s. RHS of system
    if length(nsmallind)==length(nexcind)
      temp=exp(-0.5*rlogpi);
      beta=ralpha.*temp;
      tilg=rsqpi-rtarg.*temp;
      temp3=uvec(:,:);
      klr_mexmulv(temp3,1,nn,nc,sqpi,pivec);
      rhs=temp3(nexcind)-tilg;
    else
      temp=exp(-0.5*rlogpi(nsmallind));
      beta=zeros(length(nexcind),1);
      beta(nsmallind)=ralpha(nsmallind).*temp;
      tilg=rsqpi(nsmallind)-rtarg(nsmallind).*temp;
      rrind=nexcind(nsmallind);
      temp3=uvec(:,:);
      klr_mexmulv(temp3,1,nn,nc,sqpi,pivec);
      rhs=zeros(length(nexcind),1);
      rhs(nsmallind)=temp3(rrind)-tilg;
    end
    % Diagonal of system matrix (for preconditioning)
    temp=zeros(n,1); temp(nexcind)=exp(2*rlogpi);
    if ~useworksub
      covdg=klr.covdiag+klr.bias_pvar;
    else
      covdg=klr.covdiag(subind)+klr.bias_pvar;
    end
    temp3=klr_mulp(temp.*covdg,nn,nc);
    klr_intern.diagmat=rpi.*((1-2*rpi).*covdg(nexcind)+ ...
			     temp3(nexcind))+1;
    if klr.verbose>1
      fprintf(1,'Preconditioner: max=%f, min=%f\n', ...
	      max(klr_intern.diagmat),min(klr_intern.diagmat));
    end
    % Configure and run PCG
    klr_intern.nexcind=nexcind;
    klr_intern.logpi=logpi; % Not reduced
    klr_intern.pi=pivec;
    klr_intern.sqpi=sqpi;
    [beta,flag,relres,nit]=pcg(@klr_mulsysmat,rhs,klr.cg_tol,...
			       klr.cg_maxit,@klr_divprecond,[],beta);
    if klr.verbose>1
      if flag==0
	fprintf(1,'CG: Converged to req. tol.\n');
      elseif flag==1
	fprintf(1,'CG: Run for max. number of iter.\n');
      else
	fprintf(1,'CG: Terminated for other reason (flag=%d)\n',flag);
      end
      fprintf(1,'    Number of iter.: %d\n',nit);
      fprintf(1,'    Final rel. res.: %f\n',relres);
    end
    % Compute new ALPHA from BETA. Store old ALPHA in OALPHA
    oalpha=alpha;
    if no_exc
      alpha=beta(:,:);
      klr_mexmulv(alpha,0,nn,nc,sqpi,pivec);
    else
      alpha=zeros(n,1);
      alpha(nexcind)=rsqpi.*beta;
      alpha(excind)=targ(excind);
      alpha=alpha-klr_mulp(alpha,nn,nc).*pivec;
    end
  else
    % Compute ALPHA exactly
    rhs=pivec.*uvec;
    rhs=rhs-pivec.*klr_mulp(rhs,nn,nc)-pivec+targ;
    oalpha=alpha;
    if ~usemix
      alpha=klr_solvesystem(rhs,logpi);
    else
      alpha=klr_debug_solvesystem(rhs,logpi);
    end
  end
  uvec=klr_wrap_covmul(alpha);
  klr_addsigsq(uvec,alpha,klr.bias_pvar,nn,nc);
  iter=iter+1; % Iteration done
end

if nargout>1
  varargout(1)={convflag};
  if nargout>2
    varargout(2)={logpi};
    if nargout>3
      varargout(3)={uvec};
    end
  end
end
