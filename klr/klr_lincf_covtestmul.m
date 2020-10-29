function y = klr_lincf_covtestmul(x,tedata)
%KLR_LINCF_COVTESTMUL COVTESTMUL implementation for linear kernel
%  Y = KLR_LINCF_COVTESTMUL(X,TEDATA)
%  Implements COVTESTMUL. See KLR_LINCF_COVMUL for kernel
%  representation.

global klr klr_intern;

nn=klr.num_data; nc=klr.num_class;
if size(x,1)~=nn*nc || size(x,2)~=1
  error('X has wrong size');
end
if size(tedata,2)~=size(klr_intern.xdata,2)
  error('TEDATA has wrong size');
end
nte=size(tedata,1);

fst_reshape(x,nn,nc);
if issparse(klr_intern.xdata)
  if ~issparse(tedata)
    error('TEDATA must be sparse');
  end
  y=zeros(nte,nc);
  fst_dspammt2(y,tedata,klr_intern.xdata,x);
else
  y=tedata*(klr_intern.xdata'*x);
end
vvec=exp(klr.covinfo.theta(klr.covinfo.vpar_pos));
fst_muldiag(y,vvec,0);
if isfield(klr.covinfo,'spar_pos') && ...
      ~isempty(klr.covinfo.spar_pos)
  svec=exp(klr.covinfo.theta(klr.covinfo.spar_pos));
  fst_addvec(y,svec,0);
end
fst_reshape(x,nn*nc,1);
fst_reshape(y,nte*nc,1);
