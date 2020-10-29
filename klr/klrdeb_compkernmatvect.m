function y = klrdeb_compkernmatvect(n,x,vvec,svec,mbuff,mdiag,l2p, ...
				    varargin)
%KLRDEB_COMPKERNMATVECT Debug function

indy=[]; indx=[];
ny=n; nx=n;
if nargin>7
  indy=varargin{1};
  if ~isempty(indy)
    if indy(1)==0
      ny=indy(2);
      nx=ny;
    else
      ny=length(indy); nx=ny;
      if nargin>8
	indx=varargin{2};
	if ~isempty(indx)
	  nx=length(indx);
	end
      end
    end
  end
end
[np,dummy]=size(x);
if mod(np,nx)~=0 | dummy~=1
  error('X');
end
np=np/nx;
nl=size(mdiag,2);

y=zeros(ny,np);
fst_reshape(x,nx,np);
l2ppos=1;
mpos=1; uplo='L ';
for l=1:nl
  nump=l2p(l2ppos); l2ppos=l2ppos+1;
  pind=l2p(l2ppos:(l2ppos+nump-1));
  l2ppos=l2ppos+nump;
  dgind=(0:(n-1))'*(n+1)+((mpos-1)*n+1);
  mbuff(dgind)=mdiag(:,l);
  dummy=zeros(ny,nump);
  if isempty(indy) | indy(1)==0
    fst_dsymm(dummy,{mbuff; [1; mpos; ny; ny]; uplo},x(:,pind));
  else
    if uplo(1)=='L'
      tmat=makesymm(mbuff(:,mpos:(mpos+n-1)),1);
    else
      tmat=makesymm(mbuff(:,mpos:(mpos+n-1)),0);
    end
    dummy=tmat(indy,indx)*x(:,pind);
  end
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
fst_reshape(x,nx*np,1);
fst_reshape(y,ny*np,1);
