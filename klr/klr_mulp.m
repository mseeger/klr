function y = klr_mulp(x,nn,nc,varargin)
%KLR_MULP Internal helper function
%  Y = KLR_MULP(X,NN,NC,{FLIP=0})
%  Returns Y = P*X, where P = (1 otimes I) (1' otimes I) is the
%  sum-and-distribute matrix. Note that V = (I - D*P)*D^{1/2}.
%  If FLIP==1, use P = (I otimes 1) (I otimes 1').

flip=0;
if nargin>3
  flip=varargin{1};
end
if ~flip
  fst_reshape(x,nn,nc);
  temp=sum(x,2);
  y=temp(:,ones(nc,1));
  fst_reshape(y,nn*nc,1);
  fst_reshape(x,nn*nc,1);
else
  fst_reshape(x,nn,nc);
  temp=sum(x,1);
  y=temp(ones(nn,1),:);
  fst_reshape(y,nn*nc,1);
  fst_reshape(x,nn*nc,1);
end
