function y = klr_hier_covtestmul(x,tedata)
%KLR_HIER_COVTESTMUL COVTESTMUL implementation for hierarch. classif.
%  Y = KLR_HIER_COVTESTMUL(X,TEDATA)
%  Wrapper for calls to COVPARTMUL for hierarch. classif. Requires
%  COVMMULMAT primitive in KLR.HIERARCH.COVMMULMAT.
%
%  NOTE: If the low rank option is used (KLR.HIERARCH.LOWRK.XXX),
%  we require that KLR.COVINFO.PREC_OK==1.

global klr klr_intern;

if ~isfield(klr,'hierarch') || ~klr.hierarch.use
  error('Hierarchical classification mode not active');
end
do_lowrank=(isfield(klr.hierarch,'lowrk') && ...
	    klr.hierarch.lowrk.use);
if do_lowrank && ~klr.covinfo_prec_ok
  error('Need KLR.COVINFO.PREC_OK==1 in low rank mode');
end
nc=klr.num_class; nn=klr.num_data; np=klr.hierarch.nump;
if size(x,1)~=nn*nc || size(x,2)~=1
  error('X has wrong size');
end
nte=size(tedata,1);
% Multiplication by PHI'
xtemp=zeros(nn,np);
xtemp(:,klr.hierarch.leafs)=reshape(x,nn,nc);
klr_compmatphi(xtemp,klr.hierarch.childn);
ytemp=zeros(nte,np);
% Multiplication by block-diag. tilde kernel matrix
vvec=exp(klr.covinfo.theta(klr.covinfo.vpar_pos));
use_spars=isfield(klr.covinfo,'spar_pos') && ...
	  ~isempty(klr.covinfo.spar_pos);
if use_spars
  svec=exp(klr.covinfo.theta(klr.covinfo.spar_pos));
else
  svec=zeros(np,1);
end
if isfield(klr.hierarch,'linear_kern') && klr.hierarch.linear_kern
  % Linear kernel
  if issparse(klr_intern.xdata)
    if ~issparse(tedata)
      error('TEDATA must be sparse');
    end
    fst_dspammt2(ytemp,tedata,klr_intern.xdata,xtemp);
  else
    ytemp=tedata*(klr_intern.xdata'*xtemp);
  end
  fst_muldiag(ytemp,vvec,0);
  if use_spars
    fst_addvec(ytemp,svec,0);
  end
else
  l2ppos=1;
  mpos=1;
  for l=1:size(klr_intern.covtldiag,2)
    nump=klr.hierarch.l2p(l2ppos); l2ppos=l2ppos+1;
    pind=klr.hierarch.l2p(l2ppos:(l2ppos+nump-1));
    l2ppos=l2ppos+nump;
    if ~do_lowrank
      dummy=feval(klr.hierarch.covmmulmat,xtemp(:,pind),l,tedata);
      fst_muldiag(dummy,vvec(pind),0);
      if use_spars
	fst_addvec(dummy,svec(pind),0);
      end
      ytemp(:,pind)=dummy;
    else
      szi=klr.hierarch.lowrk.actsz(l);
      perm=klr.hierarch.lowrk.perm(:,l);
      utemp=xtemp(:,pind);
      invp(perm)=(1:nn)';
      fst_permute(utemp,invp,0);
      vtemp=zeros(szi,nump);
      fst_dgemm(vtemp,{klr_intern.covmat; [1; mpos; nn; szi]},1, ...
		utemp,0);
      fst_dtrsm(vtemp,{klr_intern.covmat; [1; mpos; szi; szi]; ...
		       'L '},0,1);
      dummy=feval(klr.hierarch.covmmulmat,vtemp,l,tedata, ...
		  perm(1:szi));
      fst_muldiag(dummy,vvec(pind),0);
      if use_spars
	fst_addvec(dummy,svec(pind),0);
      end
      ytemp(:,pind)=dummy;
      mpos=mpos+szi;
    end
  end
end
% Multiplication by PHI
klr_compmatphit(ytemp,klr.hierarch.childn);
y=reshape(ytemp(:,klr.hierarch.leafs),nte*nc,1);
