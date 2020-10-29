function dummy = klr_rbfcf_comp_prec(varargin)
%KLR_RBFCF_COMP_PREC COMP_PREC implementation for RBF kernel
%  KLR_RBFCF_COMP_PREC({NEWDATA=0})
%  Computes precomp. matrix and kernel diagonal for RBF kernel
%  (see RADIALCF), to be stored in the global KLR_INTERN.COVMAT
%  and KLR.COVDIAG. The parameters THETA are read from
%  KLR.COVINFO.THETA, starting from pos. KLR.COVINFO.THPOS.
%  If KLR.COVINFO.TIED==1, THETA has the form [log(W); log(C)]
%  (see RADIALCF). Otherwise, THETA consists of these parts for
%  each kernel.
%  If KLR.COVINFO.FIXVAR given and true, C fixed to 1 and not
%  contained in THETA.
%  The RBF kernel has the form C*EXP(W*A). KLR_INTERN.PRECMAT
%  contains A, which does not depend on THETA. The kernel matrices
%  are stored in cell array KLR_INTERN.COVMAT, lower tri. for odd,
%  upper tri. for even. If KLR.COVINFO.TIED==1, KLR_INTERN.COVMAT
%  contains the full kernel matrix.
%  The data matrix (cases as rows) must be given in
%  KLR_INTERN.XDATA.

global klr klr_intern;

newdata=0;
if nargin>0
  newdata=varargin{1};
end
nn=klr.num_data; nc=klr.num_class;
if nn~=size(klr_intern.xdata,1)
  error('KLR.NUM_DATA or KLR_INTERN.XDATA wrong');
end
if isfield(klr.covinfo,'fixvar') && klr.covinfo.fixvar
  fixvar=1;
  sz=1;
else
  fixvar=0;
  sz=2;
end
if ~klr.covinfo.tied
  sz=sz*nc;
end
thpos=1;
if isfield(klr.covinfo,'thpos')
  thpos=klr.covinfo.thpos;
end
if length(klr.covinfo.theta)<(thpos+sz-1)
  error('Size of KLR.COVINFO.THETA wrong');
end
theta=klr.covinfo.theta(thpos:(thpos+sz-1));
if klr.verbose>1
  fprintf(1,'KLR_RBFCF_COMP_PREC: Doing precomputation.\n');
end

if newdata
  klr_intern.precmat=radialcf_precmat(klr_intern.xdata, ...
				      klr_intern.xdata);
else
  if size(klr_intern.precmat,1)~=nn || size(klr_intern.precmat,2)~= ...
	nn
    error('KLR_INTERN.PRECMAT has wrong size, need NEWDATA=1');
  end
end
if klr.covinfo.tied
  klr_intern.covmat=[];
  if ~fixvar
    lvar=theta(2);
  else
    lvar=0;
  end
  klr_intern.covmat=exp(exp(theta(1))*klr_intern.precmat+lvar);
  temp=exp(lvar);
  klr.covdiag=temp(ones(nn*nc,1));
else
  klr_intern.covmat=[];
  pos=1;
  if ~fixvar
    incpos=2;
  else
    incpos=1;
    lvarl=0; lvaru=0;
  end
  for i=1:floor(nc/2)
    if ~fixvar
      lvarl=theta(pos+1);
      lvaru=theta(pos+3);
    end
    klr_intern.covmat{i}=tril(exp(exp(theta(pos))* ...
				  klr_intern.precmat+lvarl), ...
			      -1)+triu(exp(exp(theta(pos+incpos))* ...
					   klr_intern.precmat+ ...
					   lvaru),1);
    pos=pos+(2*incpos);
  end
  if mod(nc,2)==1
    if ~fixvar
      lvarl=theta(pos+1);
    end
    klr_intern.covmat{floor(nc/2)+1}=exp(exp(theta(pos))* ...
					 klr_intern.precmat+ ...
					 lvarl);
  end
  if ~fixvar
    temp=exp(theta(2:2:(2*nc)));
    if size(temp,2)==1
      temp=temp';
    end
  else
    temp=ones(1,nc);
  end
  klr.covdiag=reshape(temp(ones(nn,1),:),nn*nc,1);
end
klr.covinfo.prec_ok=1;
