% Demo example for how to use a low-rank kernel matrix
% approximation obtained by incomplete Cholesky decomposition.
% NOTE: Requires CHOL_INCOMPLETE, download from my homepage

global klr klr_intern;

% Parameters
hyp_c=1;
hyp_w=9;

% Generate data
p=1500;
v=[1.5 0]; c=1;
data=[randn(p,2)-v(ones(p,1),:) c(ones(p,1))];
c=2;
data=[data; randn(p,2)+v(ones(p,1),:) c(ones(p,1))];
v=[0 -sqrt(6.75)]; c=3;
data=[data; randn(p,2)+v(ones(p,1),:) c(ones(p,1))];

klr.num_data=size(data,1); nn=klr.num_data;
klr.num_class=size(data,2); nc=klr.num_class;
n=nn*nc;
klr.covmul=@klr_covfunlr;
klr.tol=1e-6;
klr.maxiter=10;
klr.verbose=2;
klr.lpithres=-50;
klr.cg_tol=1e-5;
klr.cg_maxit=7;
klr.bias_pvar=16;
sbias=zeros(nc,1);
salpha=zeros(n,1);

% Compute low-rank approximation to kernel matrix
fprintf(1,'Computing low-rank matrix approximation\n');
hyppars=[hyp_w; hyp_c; 0];
d=45;
xdata=data(:,1:2);
dvec=sum(xdata.*xdata,2);
[lfact,pind]=chol_incomplete(nn,0,d,0,xdata,dvec,hyppars);
fprintf(1,'OK done.\n');
clear klr_intern.covfact;
clear klr_intern.covmat;
klr_intern.covfact{1}=zeros(nn,d);
klr_intern.covfact{1}(pind,:)=lfact;
temp=sum(klr_intern.covfact{1}.*klr_intern.covfact{1},2);
klr.covdiag=repmat(temp,nc,1);
% Targets
temp=zeros(n,1);
temp((0:(nn-1))'*nc+data(:,3))=1;
targ=reshape(reshape(temp,nc,nn)',n,1);

% Run KLR_FINDMAP
fprintf(1,'Running KLR_FINDMAP with low-rank matrices\n');
[alpha,bias,flag]=klr_findmap(targ,salpha,sbias);
fprintf(1,'OK done. Final BIAS:\n');
bias'

% Plot result
xrng=-4:0.2:4; yrng=-5:0.2:3;
nx=length(xrng); ny=length(yrng);
[X,Y]=meshgrid(xrng,yrng);
xtest=[reshape(X,nx*ny,1) reshape(Y,nx*ny,1)];
pred=radialcf(xtest,xdata,hyp_c,hyp_w*2)*reshape(alpha,nn,nc);
temp=bias';
pred=pred+temp(ones(nx*ny,1),:);
thres=zeros(nx*ny,nc);
for c=1:nc
  rng=[1:(c-1) (c+1):nc];
  thres(:,c)=pred(:,c)-max(pred(:,rng),[],2);
end
figure(1);
clf;
hold on;
%cols=colorcube(max([nc 8]));
for c=1:nc
  ind=find(data(:,3)==c);
  plot(xdata(ind,1),xdata(ind,2),'b.');
  Z=reshape(thres(:,c),ny,nx);
  contour(X,Y,Z,[0 0],'r-');
  pause;
end

% Compare against the full thing
klr.covmul=@klr_simplecovfun;
klr.covdiag=hyp_c(ones(n,1),:);
clear klr_intern.covfact;
clear klr_intern.covmat;
fprintf(1,'Computing full kernel matrix\n');
klr_intern.covmat{1}=radialcf(xdata,xdata,hyp_c,hyp_w*2);
fprintf(1,'OK done.\n');

% Copy and paste...
fprintf(1,'Running KLR_FINDMAP with full matrices\n');
[alpha,bias,flag]=klr_findmap(targ,salpha,sbias);
fprintf(1,'OK done. Final BIAS:\n');
bias'

% Plot result
xrng=-4:0.2:4; yrng=-5:0.2:3;
nx=length(xrng); ny=length(yrng);
[X,Y]=meshgrid(xrng,yrng);
xtest=[reshape(X,nx*ny,1) reshape(Y,nx*ny,1)];
pred=radialcf(xtest,xdata,hyp_c,hyp_w*2)*reshape(alpha,nn,nc);
temp=bias';
pred=pred+temp(ones(nx*ny,1),:);
thres=zeros(nx*ny,nc);
for c=1:nc
  rng=[1:(c-1) (c+1):nc];
  thres(:,c)=pred(:,c)-max(pred(:,rng),[],2);
end
figure(2);
clf;
hold on;
%cols=colorcube(max([nc 8]));
for c=1:nc
  Z=reshape(thres(:,c),ny,nx);
  contour(X,Y,Z,[0 0]);
  ind=find(data(:,3)==c);
  plot(xdata(ind,1),xdata(ind,2),'.');
  pause;
end
