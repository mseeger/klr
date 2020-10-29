function dummy = klr_testoptimfunc(resfname,test,maxfevals, ...
				   varargin)

% Test hyperparameter optimization

global klr klr_intern;

dooptim=1;
if nargin>3
  dooptim=varargin{1};
end

if dooptim
  nfold=length(klr.cvcrit.iind);
  for i=1:nfold
    if ~issorted(klr.cvcrit.iind{i}) || ~issorted(klr.cvcrit.jind{i})
      error('KLR.CVCRIT indexes must be sorted');
    end
  end
end

nn=klr.num_data; nc=klr.num_class; n=nn*nc;
numattr=size(klr_intern.xdata,2);
targ=klr_intern.ydata;
klr.covinfo.prec_ok=0;
salpha=zeros(n,1);
klr_intern.salpha=salpha;
klr_intern.salpha_fb=salpha; % fallback value is 0
klr_intern.best.reset=1; % reset best f-value

% Optimization
tic;
if dooptim
  if isfield(klr,'debug') && isfield(klr.debug,'optcode') && ...
	klr.debug.optcode==2
    if klr.verbose>0
      fprintf(1,'RUNNING L-BFGS...\n');
    end
    [theta,fval]=opt_BFGS('klr_critfunc',klr.covinfo.theta,'maxevals', ...
			  maxfevals,'tolerance_f',0, ...
			  'tolerance_grad',1e-8);
  else
    if klr.verbose>0
      fprintf(1,'RUNNING MINIMIZE...\n');
    end
    [theta,fvals,numit]=minimize(klr.covinfo.theta,'klr_critfunc', ...
				 -maxfevals);
  end
  fprintf(1,'\nBest crit=%f. Corr pars:\n',klr_intern.best.fval);
  exp(klr_intern.best.theta)'
  g=klr_intern.best.grad;
  fprintf(1,'Corr gradsize=%f. Grad:\n',sqrt(g'*g));
  g'
  klr.covinfo.theta=klr_intern.best.theta;
  alpha=klr_intern.best.alpha;
else
  % Compute ALPHA for given hyperpars
  if klr.verbose>0
    fprintf(1,'Running KLR_FINDMAP...\n');
  end
  [alpha,flag]=klr_findmap(targ,salpha);
  if flag==2
    error('Error running KLR_FINDMAP!');
  end
end
if klr.verbose>0
  t=toc;
  if isfield(klr,'debug') && isfield(klr.debug,'doprots') && ...
	klr.debug.doprots && isfield(klr.debug,'timeforprots')
    t=t-klr.debug.timeforprots;
  end
  fprintf(1,'\nOK: Elapsed time=%f secs\n',t);
end

% Compute ROC curve for this setup
if klr.verbose>0
  fprintf(1,'Computing ROC curve on test set...\n');
end
nte=size(test,1);
bias=klr.bias_pvar*sum(reshape(alpha,nn,nc),1);
ute=klr_wrap_covtestmul(alpha,test(:,1:numattr))+ ...
    reshape(bias(ones(nte,1),:),nte*nc,1);
lte=logsumexp(reshape(ute,nte,nc)')';
logpite=reshape(ute,nte,nc)-lte(:,ones(nc,1));
[mxval,guess]=max(logpite,[],2);
[dummy,indte]=sort(-mxval);
errte=(guess~=test(:,numattr+1));
roc=cumsum(errte(indte))/nte;
if klr.verbose>0
  fprintf(1,'Trailing values ROC:\n');
  roc((nte-4):nte)'
end

% Store stuff
theta=klr.covinfo.theta;
mvers=version;
if mvers(1)=='7'
  optstr='-v6';
else
  optstr='';
end
if isfield(klr,'debug') && isfield(klr.debug,'doprots') && ...
      klr.debug.doprots
  errte=klr.debug.prot_errte;
  loglhte=klr.debug.prot_loglhte;
  appcv=klr.debug.prot_appcv;
  truecv=klr.debug.prot_truecv;
  save(resfname,'theta','alpha','ute','roc','guess','errte', ...
       'loglhte','appcv','truecv','klr.cvcrit',optstr);
else
  if dooptim
    iind=klr.cvcrit.iind; jind=klr.cvcrit.jind;
    save(resfname,'theta','alpha','ute','roc','guess','iind', ...
	 'jind',optstr);
  else
    save(resfname,'theta','alpha','ute','roc','guess',optstr);
  end
end
