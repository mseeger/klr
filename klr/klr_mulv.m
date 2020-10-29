function y = klr_mulv(x,trs,nn,nc)
%KLR_MULV Internal helper function
%  Y = KLR_MULV(X,TRS,NN,NC)
%  Returns Y = V*X (for TRS==0) or Y = V'*X (for TRS==1). The size
%  is given by NN, NC. PI and SQRT(PI) are given in KLR_INTERN.PI
%  and KLR_INTERN.SQPI (reduced to size NN, NC)

global klr_intern;

if trs
  temp=sum(reshape(x.*klr_intern.pi,nn,nc),2);
  y=klr_intern.sqpi.*(x-reshape(temp(:,ones(nc,1)),nn*nc,1));
else
  y=klr_intern.sqpi.*x;
  temp=sum(reshape(y,nn,nc),2);
  y=y-klr_intern.pi.*reshape(temp(:,ones(nc,1)),nn*nc,1);
end
