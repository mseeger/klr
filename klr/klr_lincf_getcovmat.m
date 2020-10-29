function kmat = klr_lincf_getcovmat(c,varargin)
%KLR_LINCF_COVMULMAT COVMULMAT implementation for linear kernel
%  Y = KLR_LINCF_COVMULMAT(X)
%  Implements GETCOVMAT. See KLR_LINCF_COVMUL for kernel
%  representation.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
subind=[];
if nargin>1
  subind=varargin{1};
end
nn=klr.num_data; nc=klr.num_class;
if c<1 || c>nc
  error('C argument wrong');
end
if isempty(subind)
  kmat=(exp(klr.covinfo.theta(klr.covinfo.vpar_pos(c)))* ...
	klr_intern.xdata)*klr_intern.xdata';
  if isfield(klr.covinfo,'spar_pos') && ...
	~isempty(klr.covinfo.spar_pos);
    kmat=kmat+exp(klr.covinfo.theta(klr.covinfo.spar_pos(c)));
  end
else
  kmat=(exp(klr.covinfo.theta(klr.covinfo.vpar_pos(c)))* ...
	klr_intern.xdata(subind,:))*klr_intern.xdata(subind,:)';
  if isfield(klr.covinfo,'spar_pos') && ...
	~isempty(klr.covinfo.spar_pos);
    kmat=kmat+exp(klr.covinfo.theta(klr.covinfo.spar_pos(c)));
  end
end
