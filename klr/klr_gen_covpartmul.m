function y = klr_gen_covpartmul(x,iind,jind)
%KLR_GEN_COVPARTMUL Generic COVPARTMUL implementation
%  Y = KLR_GEN_COVPARTMUL(X,IIND,JIND)
%  Implements COVPARMUL. See KLR_GEN_COVMUL for format of kernel
%  matrix blocks in KLR_INTERN.COVMAT.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
ni=length(iind); nj=length(jind);
nc=klr.num_class;
if length(x)~=nj*nc
  error('X has wrong size');
end
y=zeros(ni*nc,1);
if ~klr.covinfo.tied
  nn=klr.num_data;
  dgind=(0:(nn-1))'*(nn+1)+1;
  lower=1; actc=0;
  for c=1:nc
    if lower
      actc=actc+1;
    end
    rngy=(ni*(c-1)+1):(ni*c);
    rngx=(nj*(c-1)+1):(nj*c);
    klr_intern.covmat{actc}(dgind)=klr.covdiag(((nn*(c-1)+1):(nn* ...
						  c))');
    y(rngy)=klr_compmatvect(klr_intern.covmat{actc},x(rngx), ...
			    double(lower),iind,jind);
    lower=~lower;
  end
else
  y=reshape(klr_intern.covmat(iind,jind)*reshape(x,nj,nc),ni*nc, ...
	    1);
end
