function y = klr_gen_covshuffle(ind,cind,flag)
%KLR_GEN_COVSHUFFLE Generic COVSHUFFLE implementation
%  Y = KLR_GEN_COVSHUFFLE(IND,CIND,FLAG)
%  Implements COVSHUFFLE, see KLR_GEN_COVMAT for format of
%  KLR_INTERN.COVMAT.
%  If KLR.COVINFO.PREC_OK==0, the function with handle
%  KLR.COMP_PREC is called before anything else.

global klr klr_intern;

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
t1=nn*(0:(nc-1));
tind=reshape(fi(:,ones(nc,1))+t1(ones(nn,1),:),nn*nc,1);
if flag
  if ~klr.covinfo.tied
    for i=1:ceil(nc/2)
      klr_intern.covmat{i}=klr_intern.covmat{i}(fi,fi);
    end
  else
    klr_intern.covmat=klr_intern.covmat(fi,fi);
  end
  % Diagonal
  klr.covdiag=klr.covdiag(tind);
else
  if ~klr.covinfo.tied
    for i=1:ceil(nc/2)
      klr_intern.covmat{i}(fi,fi)=klr_intern.covmat{i};
    end
  else
    klr_intern.covmat(fi,fi)=klr_intern.covmat;
  end
  % Diagonal
  klr.covdiag(tind)=klr.covdiag;
end
