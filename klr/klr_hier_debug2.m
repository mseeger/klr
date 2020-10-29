% Test MEX functions

maxn=400;

% COMPMATPHI, COMPMATPHIT
num=0;
for i=1:num
  % Sample tree
  done=0;
  while ~done
    childn=[4; zeros(4,1)];
    done=1;
    for p=2:11
      if p>length(childn)
	done=0; break;
      end
      if rand<(1/3)
	cn=0;
      else
	cn=2+floor(3*rand);
      end
      childn(p)=cn;
      childn=[childn; zeros(cn,1)];
    end
  end
  np=length(childn)-1;
  nn=1+floor(rand*maxn);
  x=2*rand(nn,np)-1;
  ret2=klrdeb_compmatphit(x,childn);
  ret1=x(:,:);
  klr_compmatphit(ret1,childn);
  if max(max(reldiff(ret1,ret2,1e-15)))>(1e-7)
    error('COMPMATPHIT');
  end
  ret2=klrdeb_compmatphi(x,childn);
  ret1=x(:,:);
  klr_compmatphi(ret1,childn);
  if max(max(reldiff(ret1,ret2,1e-15)))>(1e-7)
    error('COMPMATPHI');
  end
end

% COMPKERNMATMAT, COMPKERNMATVECT
num=50;
for i=1:num
  i
  n=ceil(maxn*rand);
  np=ceil(40*rand);
  nl=ceil(min([5 np])*rand);
  vvec=exp(2*rand(np,1)-1);
  svec=exp(2*rand(np,1)-1);
  mbuff=2*rand(n,n*ceil(nl/2))-1;
  mdiag=exp(2*rand(n,nl)-1);
  done=0;
  l4p=[1:nl ceil(nl*rand(1,np-nl))]';
  pm=randperm(np);
  l4p=l4p(pm);
  l2p=[];
  maxnum=0;
  for l=1:nl
    ind=find(l4p==l);
    num=length(ind); maxnum=max([maxnum num]);
    l2p=[l2p; num; ind];
  end
  % COMPKERNMATMAT
  if rand<0.5
    sz=ceil(rand*n);
  else
    sz=n;
  end
  nq=1+floor(30*rand);
  tmpbuff=zeros(2*sz*nq*maxnum,1);
  x=2*rand(sz*np,nq)-1;
  ret1=klr_compkernmatmat(n,x,vvec,svec,mbuff,mdiag,l2p,sz,tmpbuff);
  ret2=klrdeb_compkernmatmat(n,x,vvec,svec,mbuff,mdiag,l2p,sz);
  if max(max(reldiff(ret1,ret2,1e-15)))>(1e-7)
    error('COMPKERNMATMAT');
  end
  % COMPKERNMATVECT
  indx=[]; indy=[];
  nx=n; ny=n;
  u=rand;
  if u<(1/3)
    ny=ceil(rand*n); nx=ny;
    indy=[0; ny];
  elseif u<(2/3)
    ny=ceil(rand*n); nx=ceil(rand*n);
    pm=randperm(n);
    indy=pm(1:ny);
    pm=randperm(n);
    indx=pm(1:nx);
  end
  tmpbuff=zeros(2*n*maxnum,1);
  x=2*rand(nx*np,1)-1;
  ret1=klr_compkernmatvect(n,x,vvec,svec,mbuff,mdiag,l2p,indy,indx, ...
			   tmpbuff);
  ret2=klrdeb_compkernmatvect(n,x,vvec,svec,mbuff,mdiag,l2p,indy,indx);
  if max(max(reldiff(ret1,ret2,1e-15)))>(1e-7)
    error('COMPKERNMATVECT');
  end
end
