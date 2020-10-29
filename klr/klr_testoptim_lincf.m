% Test hyperparameter optimization

global klr klr_intern;

% Load data
% - Data matrix (nn-by-nd) -> KLR_INTERN.XDATA
% - Targets (nn*nc vector) -> TARG, KLR_INTERN.YDATA
klr.num_data=size(klr_intern.xdata,1);
klr.num_class=2;
nn=klr.num_data; nc=klr.num_class; n=nn*nc;
klr.verbose=1;
klr.tol=1e-11;
klr.maxiter=12;
klr.maxiter_fb=12;
klr.lpithres=-50;
klr.cg_tol=1e-12;
klr.cg_maxit=50;
klr.bias_pvar=16; % Set this to smaller value if convergence
                  % problems!
klr.cvcg_tol=1e-12;
klr.cvcg_maxit=50;

% We use 5-fold CV, drawing the permutation at random
nfold=5; nf=floor(nn/nfold);
klr.cvcrit.iind=[]; klr.cvcrit.jind=[];
pm=randperm(nn)';
for i=1:(nfold-1)
  klr.cvcrit.iind{i}=sort(pm(((i-1)*nf+1):(i*nf)));
  klr.cvcrit.jind{i}=sort(pm([1:((i-1)*nf) (i*nf+1):nn]));
end
klr.cvcrit.iind{nfold}=sort(pm(((nfold-1)*nf+1):end));
klr.cvcrit.jind{nfold}=sort(pm(1:((nfold-1)*nf)));

% Result path
fpath=strcat(getenv('HOME'),'/matlab/logregr/exp/');
numexp=1;

% Run different scenarios

% MC, APPROX.(MINIMIZE)
klr.debug.solveexact=0; % approx.
klr.debug.optcode=1; % MINIMIZE optimizer
klr.num_class=2;
klr.cg_maxit=80;
klr.cvcg_maxit=80;
% LINCF, shared v_c
klr.comp_prec=@klr_lincf_comp_prec;
klr.covmul=@klr_lincf_covmul;
klr.covpartmul=@klr_lincf_covpartmul;
klr.covshuffle=@klr_lincf_covshuffle;
klr.accum_grad=@klr_lincf_accum_grad;
klr.covtestmul=@klr_lincf_covtestmul;
fprintf(1,'\nSETTING: MC, APPROX(MINIMIZE): LINCF\n');
klr.covinfo.vpar_pos=[1 1]';
klr.covinfo.theta=0; % log of v_c
fname=strcat(fpath,'result17-',int2str(numexp),'.mat');
feval(klr.comp_prec,1); % Startup for new dataset
try
  % Run optimization
  klr_testoptimfunc(fname,test,60);
%  klr_testoptimfunc(fname,test,60,0); % NO HYPERPAR. OPTIM
catch
  err=lasterror;
  err.message
end
