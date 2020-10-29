function y = klr_lincf_covmul(x,varargin)
%KLR_LINCF_COVMUL COVMUL implementation for linear kernel
%  Y = KLR_LINCF_COVMUL(X,{IND})
%  Computes Y = K*X for the kernel matrix K. K is block-diagonal.
%  X is a vector. The blocks of K are for the linear kernel based
%  on the data matrix in KLR_INTERN.XDATA. The linear kernel
%  is K^(c) = v_c x^T y + s_c, v_c>0, s_c>=0. The log(v_c)
%  parameters are in KLR.COVINFO.THETA at pos.
%  KLR.COVINFO_VPAR_POS. The s_c pars. are used only if
%  KLR.COVINFO.SPAR_POS is not empty, in which case this contains
%  the pos. of log(s_c) in KLR.COVINFO.THETA.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
ind=[]; nn=klr.num_data;
if isfield(klr,'worksubind')
  ind=klr.worksubind;
  nn=length(ind);
end
if nargin>1
  temp=varargin{1};
  if ~isempty(temp)
    if ~isempty(ind)
      error('Cannot use argument IND together with KLR.WORKSUBIND');
    end
    ind=temp;
    if ind(1)==0
      if length(ind)~=2
	error('Invalid IND');
      end
      nn=ind(2);
    else
      [nn,b]=size(ind);
      if b~=1
	ind=ind';
	nn=b;
      end
    end
  end
end
nc=klr.num_class; n=nn*nc;
if length(x)~=n
  error('X has wrong size');
end

fst_reshape(x,nn,nc);
if issparse(klr_intern.xdata)
  y=zeros(nn,nc);
  if isempty(ind)
    fst_dspammt(y,klr_intern.xdata,x);
  elseif ind(1)==0
    fst_dspammt(y,klr_intern.xdata(1:nn,:),x);
  else
    fst_dspammt(y,klr_intern.xdata(ind,:),x);
  end
else
  if isempty(ind)
    y=klr_intern.xdata*(klr_intern.xdata'*x);
  elseif ind(1)==0
    y=klr_intern.xdata(1:nn,:)*(klr_intern.xdata(1:nn,:)'*x);
  else
    y=klr_intern.xdata(ind,:)*(klr_intern.xdata(ind,:)'*x);
  end
end
vvec=exp(klr.covinfo.theta(klr.covinfo.vpar_pos));
fst_muldiag(y,vvec,0);
if isfield(klr.covinfo,'spar_pos') && ...
      ~isempty(klr.covinfo.spar_pos)
  svec=exp(klr.covinfo.theta(klr.covinfo.spar_pos));
  fst_addvec(y,svec,0);
end
fst_reshape(x,nn*nc,1);
fst_reshape(y,nn*nc,1);
