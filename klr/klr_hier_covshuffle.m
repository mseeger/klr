function y = klr_hier_covshuffle(ind,cind,flag)
%KLR_HIER_COVSHUFFLE COVSHUFFLE implementation for hierarch. classif.
%  Y = KLR_HIER_COVSHUFFLE(IND,CIND,FLAG)
%  Implements COVSHUFFLE, see KLR_HIER_COVMUL for format of K.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

if ~isfield(klr,'hierarch') || ~klr.hierarch.use
  error('Hierarchical classification mode not active');
end
if ~klr.covinfo.prec_ok
  feval(klr.comp_prec);
end
if size(ind,2)~=1
  ind=ind';
end;
if size(cind,2)~=1
  cind=cind';
end;
nc=klr.num_class; nn=klr.num_data;
fi=[ind; cind];
invfi(fi)=(1:nn)';
if isfield(klr.hierarch,'linear_kern') && klr.hierarch.linear_kern
  % Linear kernel
  if flag
    klr_intern.xdata=klr_intern.xdata(fi,:);
  else
    klr_intern.xdata(fi,:)=klr_intern.xdata;
  end
elseif ~isfield(klr.hierarch,'lowrk') || ~klr.hierarch.lowrk.use
  if flag
    for roff=nn:nn:size(klr_intern.covmat,2)
      pt=[1; roff-nn+1; nn; nn];
      fst_permute({klr_intern.covmat; pt},invfi,1);
      fst_permute({klr_intern.covmat; pt},invfi,0);
    end
  else
    for roff=nn:nn:size(klr_intern.covmat,2)
      pt=[1; roff-nn+1; nn; nn];
      fst_permute({klr_intern.covmat; pt},fi,1);
      fst_permute({klr_intern.covmat; pt},fi,0);
    end
  end
else
  % We simply modify the P^(l) permutations here
  if flag
    % p(i) = fi(p'(i))
    for l=1:size(klr_intern.covtldiag,2)
      klr.hierarch.lowrk.perm(:,l)=invfi(klr.hierarch.lowrk.perm(:, ...
						  l));
    end
    klr.hierarch.lowrk.cmat_diag=klr.hierarch.lowrk.cmat_diag(fi,:);
  else
    % p'(i) = fi(p(i))
    for l=1:size(klr_intern.covtldiag,2)
      klr.hierarch.lowrk.perm(:,l)=fi(klr.hierarch.lowrk.perm(:, ...
						  l));
    end
    klr.hierarch.lowrk.cmat_diag(fi,:)=klr.hierarch.lowrk.cmat_diag;
  end
end
% Diagonals
t1=nn*(0:(nc-1));
tind=reshape(fi(:,ones(nc,1))+t1(ones(nn,1),:),nn*nc,1);
if flag
  fst_permute(klr_intern.covtldiag,invfi,0);
  klr.covdiag=klr.covdiag(tind);
else
  fst_permute(klr_intern.covtldiag,fi,0);
  klr.covdiag(tind)=klr.covdiag;
end
