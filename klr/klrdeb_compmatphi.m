function y = klrdeb_compmatphi(x,cn)
%KLRDEB_COMPMATPHI Debug function

[nn,np]=size(x);
if length(cn)~=np+1
  error('CN wrong size');
end
y=x;
inodes=find(cn>0);
inodes=reshape(inodes,1,length(inodes));
if inodes(1)~=1
  error('Wrong CN');
end
inodes=fliplr(inodes(2:end));
rpos=np;
for p=inodes
  % P is inner node number + 1
  cnum=cn(p);
  %[p-1 cnum rpos-cnum+1 rpos]
  y(:,p-1)=y(:,p-1)+sum(y(:,(rpos-cnum+1):rpos),2);
  rpos=rpos-cnum;
end
