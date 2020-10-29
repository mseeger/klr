function grad = klr_sqexpspeccf_accum_grad(amat,bmat)
%KLR_SQEXPSPECCF_ACCUM_GRAD Variant of KLR_SQEXPCF_ACCUM_GRAD
%  GRAD = KLR_SQEXPSPECCF_ACCUM_GRAD(AMAT,BMAT)
%  Same as KLR_SQEXPCF_ACCUM_GRAD, but with different
%  parameterization (see KLR_SQEXPSPECCF_COMP_PREC).
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
nn=klr.num_data; nc=klr.num_class; n=nn*nc;
nd=size(klr_intern.xdata,2);
[d1 nq]=size(amat); [d2 d3]=size(bmat);
if d1~=n | d2~=n | d3~=nq
  error('AMAT, BMAT have wrong size');
end
theta=klr.covinfo.theta;
grad=zeros(length(theta),1);
if ~klr.covinfo.tied
  % C parameters
  for c=1:nc
    rng=(nn*(c-1)+1):(nn*c);
    tmat=klr_intern.covmat{c}*amat(rng,:);
    grad(c*(nd+2))=sum(sum(bmat(rng,:).*tmat));
  end
  % W parameters
  for c=1:nc
    rng=(nn*(c-1)+1):(nn*c);
    off=(c-1)*(nd+2)+1;
    gcomp=0;
    for d=1:nd
      temp=klr_intern.xdata(:,d);
      atemp=muldiag(temp,amat(rng,:));
      btemp=muldiag(temp,bmat(rng,:));
      tmat=klr_intern.covmat{c}*atemp;
      tscal=sum(sum(btemp.*tmat));
      temp=temp.*temp;
      atemp=muldiag(temp,amat(rng,:));
      btemp=muldiag(temp,bmat(rng,:));
      tmat=klr_intern.covmat{c}*atemp;
      tscal=tscal-0.5*sum(sum(bmat(rng,:).*tmat));
      tmat=klr_intern.covmat{c}*amat(rng,:);
      tscal=tscal-0.5*sum(sum(btemp.*tmat));
      gcomp=gcomp+tscal;
      grad(off+d)=tscal/nd;
    end
    grad(off)=gcomp*exp(theta(off))/nd;
  end
else
  % C parameter
  grad(nd+2)=sum(sum(reshape(bmat,nn,nc*nq).*(klr_intern.covmat{1}* ...
					      reshape(amat,nn,nc* ...
						  nq))));
  % W parameter
  gcomp=0;
  for d=1:nd
    temp=klr_intern.xdata(:,d);
    temp=reshape(temp(:,ones(nc,1)),n,1);
    atemp=muldiag(temp,amat);
    btemp=muldiag(temp,bmat);
    tscal=sum(sum(reshape(btemp,nn,nc*nq).*(klr_intern.covmat{1}* ...
					    reshape(atemp,nn,nc* ...
						    nq))));
    temp=temp.*temp;
    atemp=muldiag(temp,amat);
    btemp=muldiag(temp,bmat);
    tscal=tscal-0.5*sum(sum(reshape(bmat,nn,nc*nq).* ...
			    (klr_intern.covmat{1}*reshape(atemp,nn, ...
						  nc*nq))));
    tscal=tscal-0.5*sum(sum(reshape(btemp,nn,nc*nq).* ...
			    (klr_intern.covmat{1}*reshape(amat,nn, ...
						  nc*nq))));
    gcomp=gcomp+tscal;
    grad(d+1)=tscal/nd;
  end
  grad(1)=gcomp*exp(theta(1))/nd;
end
