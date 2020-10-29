function kmat = klr_gen_getcovmat(c,varargin)
%KLR_GEN_GETCOVMAT Generic GETCOVMAT implementation
%  KMAT = KLR_GEN_GETCOVMAT(C,{SUBIND=[]})
%  Returns kernel matrix for class C in KMAT (symmetric).
%  See KLR_GEN_COVMAT for format of KLR_INTERN.COVMAT.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

subind=[];
if nargin>1
  subind=varargin{1};
end
if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
nc=klr.num_class; nn=klr.num_data;
if c<1 || c>nc
  error('C argument wrong');
end
if isempty(subind)
  if klr.covinfo.tied
    kmat=klr_intern.covmat;
  else
    actc=ceil(c/2);
    msf=(mod(c,2)==1);
    dgind=(0:(nn-1))'*(nn+1)+1;
    rng=((c-1)*nn+1):(c*nn);
    klr_intern.covmat{actc}(dgind)=klr.covdiag(rng);
    kmat=makesymm(klr_intern.covmat{actc},msf);
  end
else
  if klr.covinfo.tied
    kmat=klr_intern.covmat(subind,subind);
  else
    actc=ceil(c/2);
    msf=(mod(c,2)==1);
    dgind=(0:(nn-1))'*(nn+1)+1;
    rng=((c-1)*nn+1):(c*nn);
    klr_intern.covmat{actc}(dgind)=klr.covdiag(rng);
    kmat=makesymm(klr_intern.covmat{actc}(subind,subind),msf);
  end
end
