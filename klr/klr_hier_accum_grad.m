function grad = klr_hier_accum_grad(emat,fmat)
%KLR_HIER_ACCUM_GRAD ACCUM_GRAD implementation for hierarch. classif.
%  GRAD = KLR_HIER_ACCUM_GRAD(EMAT,FMAT)
%  Wrapper for calls to ACCUM_GRAD for hierarch. classif. See
%  KLR_HIER_COVMUL for representation of kernel matrix K.
%  Needs primitive for kernel derivative computation in
%  KLR.HIERARCH.DERIVMMULMAT (see docum. of hier. classif.).
%  In low rank mode (KLR.HIERARCH.LOWRK.USE==1), the primitives
%  KLR.HIERARCH.LOWRK.DERIVMTRACE and
%  KLR.HIERARCH.LOWRK.DERIVDGMINNER are needed instead.
%  NOTE: These primitives are used only if the correlation kernels
%  do have parameters (apart from the v_p parameters).
%  Requires KLR.COVINFO.PREC_OK==1.

global klr klr_intern;

if ~isfield(klr,'hierarch') || ~klr.hierarch.use
  error('Hierarchical classification mode not active');
end
if ~klr.covinfo.prec_ok
  error('Need KLR.COVINFO.PREC_OK==1');
end
do_lowrank=(isfield(klr.hierarch,'lowrk') && ...
	    klr.hierarch.lowrk.use);
nc=klr.num_class; nn=klr.num_data; np=klr.hierarch.nump;
nq=size(emat,2);
if size(emat,1)~=nn*nc || size(fmat,1)~=nn*nc || size(fmat,2)~=nq
  error('EMAT, FMAT wrong');
end
grad=zeros(length(klr.covinfo.theta),1);
nl=length(klr.hierarch.mpar_num);
% Compute \tilde{E}, \tilde{F} matrices
tlemat=zeros(nn*nq,np);
dummy=emat(:,:); % force true copy
fst_flipdims(dummy,nn);
tlemat(:,klr.hierarch.leafs)=dummy;
klr_compmatphi(tlemat,klr.hierarch.childn);
tlfmat=zeros(nn*nq,np);
dummy=fmat(:,:); % force true copy
fst_flipdims(dummy,nn);
tlfmat(:,klr.hierarch.leafs)=dummy;
klr_compmatphi(tlfmat,klr.hierarch.childn);
vvec=reshape(exp(klr.covinfo.theta(klr.covinfo.vpar_pos)),np,1);
if isfield(klr.covinfo,'spar_pos') && ~isempty(klr.covinfo.spar_pos)
  svec=reshape(exp(klr.covinfo.theta(klr.covinfo.spar_pos)),np,1);
  % Gradient for s_p parameters
  tvec=fst_diagmul(tlemat,tlfmat,0).*svec;
  grad=fst_sumpos(tvec,klr.covinfo.spar_pos,length(grad));
end
if isfield(klr.hierarch,'linear_kern') && klr.hierarch.linear_kern
  % Linear kernel
  % Has v_p parameters only
  df=size(klr_intern.xdata,2);
  fst_reshape(tlfmat,nn,nq*np);
  if df<nn
    if issparse(klr_intern.xdata)
      dummy=zeros(df,nq*np);
      fst_dspamm(dummy,klr_intern.xdata,tlfmat,1);
      %dummy=full(klr_intern.xdata'*tlfmat);
      dummy2=zeros(df,nq*np);
      fst_dspamm(dummy2,klr_intern.xdata,tlemat,1);
      %dummy2=full(klr_intern.xdata'*tlemat);
    else
      dummy=klr_intern.xdata'*tlfmat;
      dummy2=klr_intern.xdata'*tlemat;
    end
    fst_reshape(dummy,df*nq,np);
    fst_reshape(dummy2,df*nq,np);
    tvec=fst_diagmul(dummy,dummy2,0).*vvec;
  else
    if issparse(klr_intern.xdata)
      dummy=zeros(nn,nq*np);
      fst_dspammt(dummy,klr_intern.xdata,tlfmat);
      %dummy=full(klr_intern.xdata*full(klr_intern.xdata'*tlfmat));
    else
      dummy=klr_intern.xdata*(klr_intern.xdata'*tlfmat);
    end
    fst_reshape(dummy,nn*nq,np);
    tvec=fst_diagmul(tlemat,dummy,0).*vvec;
  end
  grad=grad+fst_sumpos(tvec,klr.covinfo.vpar_pos,length(grad));
else
  % General case
  % Main loop over l
  l2ppos=1; mpos=1; uplo='L ';
  for l=1:nl
    nump=klr.hierarch.l2p(l2ppos); l2ppos=l2ppos+1;
    pind=klr.hierarch.l2p(l2ppos:(l2ppos+nump-1));
    l2ppos=l2ppos+nump;
    % Assemble tilde stuff
    if nl>1
      tempe=tlemat(:,pind);
      tempf=tlfmat(:,pind);
    else
      % Avoids the copy if there is only a single M^(l)
      tempe=tlemat;
      tempf=tlfmat;
    end
    fst_reshape(tempf,nn,nq*nump);
    % v_p parameters
    if ~do_lowrank
      dgind=(0:(nn-1))'*(nn+1)+((mpos-1)*nn+1);
      klr_intern.covmat(dgind)=klr_intern.covtldiag(:,l);
      dummy=zeros(nn,nq*nump);
      fst_dsymm(dummy,{klr_intern.covmat; [1; mpos; nn; nn]; uplo}, ...
		tempf);
      fst_reshape(dummy,nn*nq,nump);
      tvec=fst_diagmul(tempe,dummy,0).*vvec(pind);
    else
      % Low rank mode
      szi=klr.hierarch.lowrk.actsz(l);
      perm=klr.hierarch.lowrk.perm(:,l);
      tmat=zeros(szi,nq*nump);
      fst_dgemm(tmat,{klr_intern.covmat; [1; mpos; nn; szi]},1, ...
		tempf,0);
      dummy=zeros(nn,nq*nump);
      fst_dgemm(dummy,{klr_intern.covmat; [1; mpos; nn; szi]},0, ...
		tmat,0);
      fst_permute(dummy,perm,0);
      dummy=dummy+muldiag(klr_intern.covtldiag(:,l)- ...
			  klr.hierarch.lowrk.cmat_diag(:,l),tempf);
      fst_reshape(dummy,nn*nq,nump);
      tvec=fst_diagmul(tempe,dummy,0).*vvec(pind);
    end
    grad=grad+fst_sumpos(tvec,klr.covinfo.vpar_pos(pind), ...
			 length(grad));
    % M^(l) parameters
    if klr.hierarch.mpar_num(l)>0
      if ~do_lowrank
	for ppos=1:klr.hierarch.mpar_num(l)
	  dummy=feval(klr.hierarch.derivmmulmat,tempf,l,ppos);
	  fst_reshape(dummy,nn*nq,nump);
	  tvec=fst_diagmul(tempe,dummy,0);
	  grad(klr.hierarch.mpar_pos(l)+ppos-1)=tvec'*vvec(pind);
	end
	% Increase counters
	if uplo(1)=='L'
	  uplo(1)='U';
	else
	  uplo(1)='L';
	  mpos=mpos+nn;
	end
      else
	% Low rank mode
	% Precomputations
	fst_reshape(tempe,nn,nq*nump);
	fst_dtrsm({klr_intern.covmat; [1; mpos; nn; szi]}, ...
		  {klr_intern.covmat; [1; mpos; szi; szi]; 'L '},0, ...
		  0);
	fst_permute({klr_intern.covmat; [1; mpos; nn; szi]},perm, ...
		    0);
	% We need a copy of R here:
	dummy=klr_intern.covmat(:,mpos:(mpos+szi-1)); % accumulator
	tvec=vvec(pind)';
	pivec=reshape(tvec(ones(nq,1),:),nq*nump,1);
	evec=fst_diagmul(tempf,tempe,1,pivec);
	tildemat=zeros(szi,nq*nump);
	fst_dgemm(tildemat,dummy,1,tempe,0);
	tildfmat=zeros(szi,nq*nump);
	fst_dgemm(tildfmat,dummy,1,tempf,0);
	tvec=zeros(nn,1);
	tvec(perm)=-2*evec; % -2 P e
	fst_muldiag(dummy,tvec,1);
	smmat=zeros(szi,szi);
	fst_dgemm(smmat,{klr_intern.covmat; [1; mpos; nn; szi]},1, ...
		  dummy,0,-0.5);
	fst_muldiag(tildemat,pivec,0);
	iind=perm(1:szi); % active set index
	dummy(iind,:)=dummy(iind,:)+(smmat-tildemat*tildfmat');
	fst_muldiag(tildfmat,pivec,0);
	fst_dgemm(dummy,tempe,0,tildfmat,1,1,1);
	fst_dgemm(dummy,tempf,0,tildemat,1,1,1);
	% Gradient accumulation
	for ppos=1:klr.hierarch.mpar_num(l)
	  grad(klr.hierarch_mpar_pos(l)+ppos-1)= ...
	      feval(klr.hierarch.lowrk.derivdgminner,evec,l,ppos)+ ...
	      feval(klr.hierarch.lowrk.derivmtrace,dummy,l,ppos, ...
		    iind);
	end
	% Increase counters
	mpos=mpos+szi;
      end
    end
  end
end
