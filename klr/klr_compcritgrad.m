function [crit,varargout] = klr_compcritgrad(targ,alpha,logpi,uvec, ...
					     iind,jind)
%KLR_COMPCRITGRAD Compute cross-validation criterion (and gradient)
%  [CRIT,{GRAD}] = KLR_COMPCRITGRAD(TARG,ALPHA,LOGPI,UVEC,IIND,
%                  JIND)
%  Computes cross-validation criterion and returns in CRIT. If GRAD
%  is given, the gradient w.r.t. kernel parameters
%  KLR.COVINFO.THETA is computed and returned in GRAD. ALPHA
%  are the current primary parameters, current LOG(PI) and U
%  given in LOGPI, UVEC. TARG are the soft targets. IIND and
%  JIND are cell arrays of the same size, containing index sets I,
%  J. For each pair, J must be the complement of I. The sets I in
%  IIND must be a partition of 1..KLR.NUM_DATA (not checked!). We
%  need fields of the global structure KLR as required by
%  KLR_ACCUMCRITGRAD. Especially:
%  - KLR.COVMUL:     Function object for X -> K*X, K the kernel
%    matrix
%  - KLR.COVPARTMUL: Function object for X -> K(I,J)*X, where I, J
%    are subindexes
%  - KLR.COMP_PREC:  Function object required by the other ones
%  - KLR.ACCUM_GRAD: Function object for gradient accumulation,
%    given matrices AMAT, BMAT
%  This function calls KLR_ACCUMCRITGRAD for each I and sums the
%  results for CRIT. If the gradient is to be computed, the
%  accumulation matrices are built in KLR_INTERN.ACCUM_AMAT and
%  KLR_INTERN.ACCUM_BMAT, then KLR.ACCUM_GRAD is called to compute
%  the gradient.

global klr klr_intern;

% Initialization. Sanity checks
if nargout>2
  error('Too many output arguments');
end
dograd=(nargout>1);
nfold=length(iind);
if length(jind)~=nfold
  error('IIND, JIND wrong');
end

nn=klr.num_data; nc=klr.num_class; n=nn*nc;
if dograd
  klr_intern.accum_amat=[];
  klr_intern.accum_bmat=[];
end
crit=0;
solveexact=isfield(klr,'debug') && isfield(klr.debug,'solveexact') && ...
    klr.debug.solveexact;
if isfield(klr,'mixmat') && isfield(klr.mixmat,'use') && ...
      klr.mixmat.use
  usemix=1;
else
  usemix=0;
end
if solveexact && ~usemix
  % Precomputation for exact solutions
  dummy=klr_solvesystem(zeros(n,1),logpi);
end
% Part for penalty (linear system for S4VEC)
if isfield(klr,'debug') && isfield(klr.debug,'addpenalty') && ...
      klr.debug.addpenalty
  crit=0.5*uvec'*alpha;
  if dograd
    % Compute S4VEC
    if solveexact
      % Solve system exactly
      if ~usemix
	s4vec=klr_solvesystem(alpha,logpi,1)-0.5*alpha;
      else
	s4vec=klr_debug_solvesystem(alpha,logpi,1)-0.5*alpha;
      end
    else
      % Solve system approximately
      % Special treatment for numerical stability
      pivec=exp(logpi); sqpi=exp(0.5*logpi);
      temp=(logpi<klr.lpithres)&(targ>0);
      excind=find(temp); nexcind=find(1-temp);
      if klr.verbose>1 && ~isempty(excind)
	fprintf(1,'PI too small on %d components, treated specially.\n',...
		length(excind));
      end
      rlogpi=logpi(nexcind); rtarg=targ(nexcind);
      ralpha=alpha(nexcind);
      rpi=pivec(nexcind); rsqpi=sqpi(nexcind);
      rhs=ralpha./rsqpi;
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
      % Configure and run PCG
      klr_intern.nexcind=nexcind;
      klr_intern.logpi=logpi; % Not reduced
      klr_intern.pi=pivec;
      klr_intern.sqpi=sqpi;
      beta=zeros(length(nexcind),1); % start from 0
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
      [beta,flag,relres,nit]=pcg(@klr_mulsysmat,rhs,cgtol,...
				 cgmaxit,@klr_divprecond,[],beta);
      if klr.verbose>1
	if flag==0
	  fprintf(1,'CG(S4VEC): Converged to req. tol.\n');
	elseif flag==1
	  fprintf(1,'CG(S4VEC): Run for max. number of iter.\n');
	else
	  fprintf(1,'CG(S4VEC): Terminated for other reason (flag=%d)\n',flag);
	end
	fprintf(1,'    Number of iter.: %d\n',nit);
	fprintf(1,'    Final rel. res.: %f\n',relres);
      end
      % Compute S4VEC from BETA
      temp=zeros(n,1);
      temp(nexcind)=rsqpi.*beta;
      temp(excind)=rtarg;
      s4vec=temp-klr_mulp(temp,nn,nc).*exp(logpi);
    end
    % Contribution to gradient accus
    klr_intern.accum_amat=alpha;
    klr_intern.accum_bmat=s4vec;
  end
end

% Loop over folds
for i=1:nfold
  crit=klr_accumcritgrad(targ,alpha,logpi,iind{i},jind{i},crit, ...
			 ~dograd);
end
crit=crit/nn;
if dograd
  grad=klr_wrap_accum_grad(klr_intern.accum_amat, ...
			   klr_intern.accum_bmat)/nn;
  varargout(1)={grad};
end

% Evaluate protocol scores
if isfield(klr,'debug') && isfield(klr.debug,'doprots') && ...
      klr.debug.doprots
  if isfield(klr.debug,'timeforprots')
    tbegin=toc;
  end
  if isfield(klr.debug,'prot_appcv')
    % Our CV approx.
    klr.debug.prot_appcv=[klr.debug.prot_appcv; crit];
  end
  if isfield(klr.debug,'prot_errte')
    % Compute test error (test set in KLR.DEBUG.TESTX,
    % KLR.DEBUG.TESTY
    nte=size(klr.debug.testx,1);
    bias=klr.bias_pvar*sum(reshape(alpha,nn,nc),1);
    ute=klr_wrap_covtestmul(alpha,klr.debug.testx)+ ...
	reshape(bias(ones(nte,1),:),nte*nc,1);
    lte=logsumexp(reshape(ute,nte,nc)')';
    logpite=reshape(ute,nte,nc)-lte(:,ones(nc,1));
    [dummy,guess]=max(logpite,[],2);
    errte=sum(guess~=klr.debug.testy)/nte;
    klr.debug.prot_errte=[klr.debug.prot_errte; errte];
    if klr.verbose>0
      fprintf(1,'COMPCRITGRAD: Test error=%f\n',errte);
    end
    loglhte=sum(logpite((1:nte)'+nte*(klr.debug.testy-1)))/nte;
    klr.debug.prot_loglhte=[klr.debug.prot_loglhte; loglhte];
    if klr.verbose>0
      fprintf(1,'COMPCRITGRAD: Test log lh=%f\n',loglhte);
    end
  end
  if isfield(klr.debug,'prot_truecv')
    % Compute true CV score
    if klr.verbose>0
      fprintf(1,'COMPCRITGRAD: Compute true CV score...\n');
    end
    truecv=0;
    for i=1:nfold
      klr.worksubind=jind{i};
      nnj=length(jind{i});
      temp=nn*(0:(nc-1));
      jcind=reshape(jind{i}(:,ones(nc,1))+temp(ones(nnj,1),:),nnj*nc, ...
		    1);
      if klr.verbose>0
	fprintf(1,'Running KLR_FINDMAP...\n');
      end
      oldmi=klr.maxiter;
      if isfield(klr,'maxiter_fb')
	klr.maxiter=klr.maxiter_fb;
      end
      [alphaj,flag]=klr_findmap(targ(jcind),zeros(nnj*nc,1));
      klr.maxiter=oldmi;
      if flag==2
	if klr.verbose>0
	  fprintf(1,'COMPCRITGRAD: Error!! Prot. value will be messed up!!\n');
	end
      end
      uip=klr_tilcovpartmul(alphaj,iind{i},jind{i});
      nni=length(iind{i});
      lip=logsumexp(reshape(uip,nni,nc)')';
      logpiip=uip-reshape(lip(:,ones(nc,1)),nni*nc,1);
      icind=reshape(iind{i}(:,ones(nc,1))+temp(ones(nni,1),:),nni*nc, ...
		    1);
      truecv=truecv-targ(icind)'*uip+sum(lip);
    end
    truecv=truecv/nn;
    klr.worksubind=[];
    klr.debug.prot_truecv=[klr.debug.prot_truecv; truecv];
    if klr.verbose>0
      fprintf(1,'COMPCRITGRAD: True CV score=%f\n',truecv);
    end
  end
  if isfield(klr.debug,'timeforprots')
    tend=toc;
    klr.debug.timeforprots=klr.debug.timeforprots+(tend-tbegin);
  end
end

% Protocol score for hierarchical mode
if isfield(klr,'hierarch') && isfield(klr.hierarch,'doprots') && ...
      klr.hierarch.doprots
  t_start=toc;
  if klr.verbose>0
    fprintf(1,'KLR_COMPCRITGRAD: Evaluate predictive test scores.\n');
  end
  [t_acc,t_prec,t_taxo,t_pacc,tp_acc,tp_taxo,tp_pacc] = ...
      klr_hier_evaltest(alpha,klr.debug.testx,klr.debug.testy);
  if isfield(klr.debug,'do_time') && klr.debug.do_time
    fprintf(1,'Time (HIER_EVALTEST): %f secs\n',toc-t_start);
  end
  if klr.verbose>0
    fprintf(1,'Test scores:\nAccuracy: %f\n',t_acc);
    fprintf(1,'Expected accuracy: %f\n',tp_acc);
    fprintf(1,'Precision: %f\n',t_prec);
    fprintf(1,'Taxo loss: %f\n',t_taxo);
    fprintf(1,'Expected taxo loss: %f\n',tp_taxo);
    fprintf(1,'Parent accuracy: %f\n',t_pacc);
    fprintf(1,'Expected parent accuracy: %f\n',tp_pacc);
  end
  klr.debug.prot_acc=[klr.debug.prot_acc; t_acc];
  klr.debug.prot_prec=[klr.debug.prot_prec; t_prec];
  klr.debug.prot_taxo=[klr.debug.prot_taxo; t_taxo];
  klr.debug.prot_pacc=[klr.debug.prot_pacc; t_pacc];
  klr.debug.prot_eacc=[klr.debug.prot_eacc; tp_acc];
  klr.debug.prot_etaxo=[klr.debug.prot_etaxo; tp_taxo];
  klr.debug.prot_epacc=[klr.debug.prot_epacc; tp_pacc];
  % Store protocol logs intermediately
  if isfield(klr.debug,'saveprots') && ~ ...
	isempty(klr.debug.saveprots)
    if isfield(klr.debug,'prot_appcv')
      appcv=klr.debug.prot_appcv;
      sappcv='appcv';
    else
      sappcv='';
    end
    acc=klr.debug.prot_acc;
    prec=klr.debug.prot_prec;
    taxo=klr.debug.prot_taxo;
    pacc=klr.debug.prot_pacc;
    eacc=klr.debug.prot_eacc;
    etaxo=klr.debug.prot_etaxo;
    epacc=klr.debug.prot_epacc;
    save(klr.debug.saveprots,'acc','prec','taxo','pacc','eacc', ...
	 'etaxo','epacc',sappcv);
  end
end
