function y = klrdeb_compmatphit(x,cn)
%KLRDEB_COMPMATPHIT Debug function

[nn,np]=size(x);
if length(cn)~=np+1
  error('CN wrong size');
end
y=x;
inodes=find(cn>0);
if inodes(1)~=1
  error('Wrong CN');
end
inodes=inodes(2:end);
lpos=cn(1);
for p=reshape(inodes,1,length(inodes))
  % P is inner node number + 1
  cnum=cn(p);
  tvec=y(:,p-1);
  y(:,(lpos+1):(lpos+cnum))=y(:,(lpos+1):(lpos+cnum))+tvec(:, ...
						  ones(cnum,1));
  lpos=lpos+cnum;
end
