% Test hyperparameter optimization

global klr klr_intern;

% Load data
load('satimage2.mat');
klr.num_data=size(train,1);
klr.num_class=6;
nn=klr.num_data; nc=klr.num_class; n=nn*nc;
numattr=size(train,2)-1;
temp=zeros(n,1);
temp((0:(nn-1))'*nc+train(:,numattr+1))=1;
targ=reshape(reshape(temp,nc,nn)',n,1);
klr_intern.xdata=train(:,1:numattr);
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
klr.num_class=6;
klr_intern.ydata=targ;
klr.cg_maxit=80;
klr.cvcg_maxit=80;
% RBF, different
klr.covinfo.tied=0;
klr.comp_prec=@klr_rbfcf_comp_prec;
klr.covmul=@klr_gen_covmul;
klr.covpartmul=@klr_gen_covpartmul;
klr.covshuffle=@klr_gen_covshuffle;
klr.accum_grad=@klr_rbfcf_accum_grad;
klr.covtestmul=@klr_rbfcf_covtestmul;
fprintf(1,'\nSETTING: MC, APPROX(MINIMIZE): RBFCF, different\n');
temp=log([0.017 10])';
klr.covinfo.theta=reshape(temp(:,ones(nc,1)),2*nc,1);
fname=strcat(fpath,'result17-',int2str(numexp),'.mat');
%klr.debug.doprots=1;
%klr.debug.prot_errte=[];
%klr.debug.prot_loglhte=[];
%klr.debug.prot_appcv=[];
%klr.debug.prot_truecv=[];
%klr.debug.timeforprots=0;
feval(klr.comp_prec,1); % Startup for new dataset
try
  % Run optimization
  klr_testoptimfunc(fname,test,60);
catch
  err=lasterror;
  err.message
end

% OREST, APPROX
klr.debug.solveexact=0; % approx.
klr.debug.optcode=1; % MINIMIZE optimizer
klr.num_class=2;
klr.cg_maxit=40;
klr.cvcg_maxit=40;
% RBF, tied
klr.covinfo.tied=1;
fprintf(1,'\nSETTING: OREST, APPROX(MINIMIZE): RBFCF\n');
for c=1:nc
  fprintf(1,'Class %d against rest\n',c);
  klr.covinfo.theta=log([0.017 10])';
  fname=strcat(fpath,'onerest13-',int2str(numexp),'-',int2str(c), ...
	       '.mat');
  ind=(((c-1)*nn+1):(c*nn))';
  temp=targ(ind);
  klr_intern.ydata=reshape([temp 1-temp],2*nn,1);
  try
    klr_testoptimfunc_1rest(c,fname,test,60);
  catch
    err=lasterror;
    err.message
  end
end
