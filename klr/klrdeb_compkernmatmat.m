function y = klrdeb_compkernmatmat(n,x,vvec,svec,mbuff,mdiag,l2p, ...
				   varargin)
%KLRDEB_COMPKERNMATVECT Debug function

sz=n;
if nargin>7
  sz=varargin{1};
end
[np,nq]=size(x);
if mod(np,sz)~=0
  error('X');
end
np=np/sz;
nl=size(mdiag,2);

y=zeros(sz*nq,np);
xtemp=x(:,:);
fst_flipdims(xtemp,sz);
l2ppos=1;
mpos=1; uplo='L ';
for l=1:nl
  nump=l2p(l2ppos); l2ppos=l2ppos+1;
  pind=l2p(l2ppos:(l2ppos+nump-1));
  l2ppos=l2ppos+nump;
  dgind=(0:(n-1))'*(n+1)+((mpos-1)*n+1);
  mbuff(dgind)=mdiag(:,l);
  dummy=zeros(sz,nq*nump);
  fst_dsymm(dummy,{mbuff; [1; mpos; sz; sz]; uplo}, ...
	    reshape(xtemp(:,pind),sz,nq*nump));
  fst_reshape(dummy,sz*nq,nump);
  fst_muldiag(dummy,vvec(pind),0);
  fst_addvec(dummy,svec(pind),0);
  y(:,pind)=dummy;
  if uplo(1)=='L'
    uplo(1)='U';
  else
    uplo(1)='L';
    mpos=mpos+n;
  end
end
fst_flipdims(y,sz);
