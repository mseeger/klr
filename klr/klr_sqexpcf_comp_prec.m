function dummy = klr_sqexpcf_comp_prec(varargin)
%KLR_SQEXPCF_COMP_PREC COMP_PREC implementation for SQEXP kernel
%  KLR_SQEXPCF_COMP_PREC({NEWDATA=0})
%  Computes precomp. matrix and kernel diagonal for SQEXP kernel
%  (see SQEXPCF), to be stored in the global KLR_INTERN.COVMAT
%  and KLR.COVDIAG. Uses the same repres. as KLR_RBFCF_COMP_PREC
%  (but no KLR_INTERN.PRECMAT).
%  If KLR.COVINFO.TIED==1, THETA has the form [log(W); log(C)]
%  (see SQEXPCF, W is a D-vector). Otherwise, THETA consists of
%  these parts for each kernel.
%  The data matrix (cases as rows) must be given in
%  KLR_INTERN.XDATA.

global klr klr_intern;

nn=klr.num_data; nc=klr.num_class;
if nn~=size(klr_intern.xdata,1)
  error('KLR.NUM_DATA or KLR_INTERN.XDATA wrong');
end
nd=size(klr_intern.xdata,2);
if isfield(klr.covinfo,'fixvar') && klr.covinfo.fixvar
  fixvar=1;
  sz=nd;
else
  fixvar=0;
  sz=nd+1;
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
  fprintf(1,'KLR_SQEXPCF_COMP_PREC: Doing precomputation.\n');
end

if klr.covinfo.tied
  klr_intern.covmat=[];
  if ~fixvar
    var=exp(theta(nd+1));
  else
    var=1;
  end
  klr_intern.covmat=sqexpcf(klr_intern.xdata,klr_intern.xdata, ...
			    var,exp(theta(1:nd)));
  klr.covdiag=var(ones(nn*nc,1));
else
  klr_intern.covmat=[];
  pos=1;
  if ~fixvar
    incpos=nd+1;
  else
    incpos=nd;
    varl=1; varu=1;
  end
  for i=1:floor(nc/2)
    ipos=pos+incpos;
    if ~fixvar
      varl=exp(theta(pos+nd));
      varu=exp(theta(ipos+nd));
    end
    tmat1=sqexpcf(klr_intern.xdata,klr_intern.xdata, ...
		  varl,exp(theta(pos:(pos+nd-1))));
    tmat2=sqexpcf(klr_intern.xdata,klr_intern.xdata, ...
		  varu,exp(theta(ipos:(ipos+nd-1))));
    klr_intern.covmat{i}=tril(tmat1,-1)+triu(tmat2,1);
    pos=ipos+incpos;
  end
  if mod(nc,2)==1
    if ~fixvar
      varl=exp(theta(pos+nd));
    end
    klr_intern.covmat{floor(nc/2)+1}=sqexpcf(klr_intern.xdata, ...
					     klr_intern.xdata,varl, ...
					     exp(theta(pos:(pos+nd- ...
						  1))));
  end
  if ~fixvar
    temp=exp(theta((nd+1):(nd+1):(nc*(nd+1))));
    if size(temp,2)==1
      temp=temp';
    end
  else
    temp=ones(1,nc);
  end
  klr.covdiag=reshape(temp(ones(nn,1),:),nn*nc,1);
end
klr.covinfo.prec_ok=1;
