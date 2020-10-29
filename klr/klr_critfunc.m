function [fval,grad] = klr_critfunc(theta)
%KLR_CRITFUNC Criterion function for hyperpar. optimization
%  [FVAL,GRAD] = KLR_CRITFUNC(THETA)
%  Computes function value and gradient for CV criterion at
%  parameters THETA. Further arguments have to be passed via the
%  global structures KLR and KLR_INTERN. Starting values for ALPHA
%  have to be given in KLR_INTERN.SALPHA.
%  These are reoptimized first for parameters THETA (which are
%  written to KLR.COVINFO.THETA), the result is written back into
%  KLR_INTERN.SALPHA. If the reoptimization fails
%  starting from these values, it is done again starting from the
%  values KLR_INTERN.SALPHA_FB (these fall-back values are never
%  changed).
%  The soft targets must be in KLR_INTERN.YDATA. The CV partitions
%  must be given in KLR.CVCRIT.IIND, KLR.CVCRIT.JIND (see
%  KLR_COMPCRITGRAD).
%  The smallest crit. value and corr. THETA, gradient value is
%  maintained in KLR_INTERN.BEST.FVAL, KLR_INTERN.BEST.THETA,
%  KLR_INTERN.BEST.GRAD. This function is reset if
%  KLR_INTERN.BEST.RESET==1.


global klr klr_intern;

% Set new parameters. Run KLR_FINDMAP for new ALPHA, BIAS
klr.covinfo.theta=theta;
klr.covinfo.prec_ok=0;
targ=klr_intern.ydata;
alpha=klr_intern.salpha;
nn=klr.num_data; nc=klr.num_class;
if klr.verbose>0
  fprintf(1,'\nCRITFUNC: Evaluate at:\n');
  if ~isfield(klr,'mixmat') || ~isfield(klr.mixmat,'use') ||  ...
	klr.mixmat.use~=2
    if length(theta)>1
      exp(theta)'
    else
      fprintf(1,'%f\n',exp(theta));
    end
  else
    i=length(theta);
    exp(theta(1:(i-nc*nc)))'
    reshape(theta((i-nc*nc+1):i),nc,nc)
  end
end
do_time=isfield(klr,'debug') && isfield(klr.debug,'do_time') && ...
	klr.debug.do_time;
trustit=1;
iter=0;
oldmaxiter=klr.maxiter;
while iter<2
  if klr.verbose>0
    fprintf(1,'Running KLR_FINDMAP...\n');
  end
  if do_time
    start_t=toc;
  end
  [alpha,flag,logpi,uvec]=klr_findmap(targ,alpha);
  if do_time
    fprintf(1,'Time (FINDMAP): %f secs\n',toc-start_t);
  end
  if flag==2
    if klr.verbose>0 && iter==0
      fprintf(1,'\nCRITFUNC: Error! Try again from fallback values\n');
    end
    alpha=klr_intern.salpha_fb;
    iter=iter+1;
    if isfield(klr,'maxiter_fb')
      klr.maxiter=klr.maxiter_fb; % allow for different max. iter.
    end
    continue;
  else
    if klr.verbose>0
      fprintf(1,'\nCRITFUNC: OK, flag=%d. Storing new ALPHA\n',flag);
    end
    klr_intern.salpha=alpha;
    break;
  end
end
klr.maxiter=oldmaxiter;
if iter==2
  % ATTENTION: Would be better to exit with NaN here!
  if klr.verbose>0
    fprintf(1,'CRITFUNC: Critical! Cannot compute ALPHA!\n');
  end
  alpha=klr_intern.salpha;
  trustit=0;
end

% Return with NaN if TRUSTIT==0
if ~trustit
  if klr.verbose>0
    fprintf(1,'CRITFUNC: Return NaN to optimizer.\n');
  end
  fval=NaN; grad=NaN;
  return;
end

% Compute criterion, gradient
if klr.verbose>0
  fprintf(1,'CRITFUNC: Computing criterion, gradient\n');
end
if do_time
  start_t=toc;
end
[fval,grad]=klr_compcritgrad(targ,alpha,logpi,uvec, ...
			     klr.cvcrit.iind,klr.cvcrit.jind);
if do_time
  fprintf(1,'Time (COMPCRITGRAD): %f secs\n',toc-start_t);
end
if isfield(klr.covinfo,'hyperprior')
  % Hyperprior contribution (normalize by NN)
  [lprval,lprgrad]=feval(klr.covinfo.hyperprior);
  fval=fval+lprval/nn;
  grad=grad+lprgrad/nn;
end
if klr.verbose>0
  fprintf(1,'\nCRITFUNC: OK: crit=%f, size(grad)=%f\n',fval, ...
	  sqrt(grad'*grad));
end
if trustit && (klr_intern.best.reset || fval<klr_intern.best.fval)
  klr_intern.best.reset=0;
  klr_intern.best.fval=fval;
  klr_intern.best.theta=theta;
  klr_intern.best.grad=grad;
  klr_intern.best.alpha=alpha;
  if isfield(klr_intern.best,'fname')
    if klr.verbose>0
      fprintf(1,'Saving new best parameters.\n');
    end
    save(klr_intern.best.fname,'fval','theta','grad','alpha');
  end
end
