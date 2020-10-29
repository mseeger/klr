function dummy = klr_hier_compcovdiag
%KLR_HIER_COMPCOVDIAG Computes KLR.COVDIAG for hierarch. classif.
%  KLR_HIER_COMPCOVDIAG
%  Computes diagonal of K and writes into KLR.COVDIAG. Called from
%  within COMP_PREC implementation, the repres. for K must already
%  have been computed.

global klr klr_intern;

if ~isfield(klr,'hierarch') || ~klr.hierarch.use
  error('Hierarchical classification mode not active');
end
nn=klr.num_data;
nc=klr.num_class; np=klr.hierarch.nump;
vvec=exp(klr.covinfo.theta(klr.covinfo.vpar_pos));
if isfield(klr.covinfo,'spar_pos') && ~isempty(klr.covinfo.spar_pos)
  svec=exp(klr.covinfo.theta(klr.covinfo.spar_pos));
else
  svec=zeros(np,1);
end
dmat=zeros(nn,np+1); % first col. for p=0
cs=0;
inodes=find(klr.hierarch.childn>0);
for p=reshape(inodes,1,length(inodes))
  % P is inner node number + 1
  cn=klr.hierarch.childn(p);
  tvec=dmat(:,p); ti=((cs+1):(cs+cn))';
  spvec=reshape(svec(ti),1,cn);
  dmat(:,ti+1)=tvec(:,ones(cn,1))+ ...
      muldiag(klr_intern.covtldiag(:,klr.hierarch.p2l(ti)), ...
	      vvec(ti))+spvec(ones(nn,1),:);
  cs=cs+cn;
end
klr.covdiag=reshape(dmat(:,klr.hierarch.leafs+1),nn*nc,1);
