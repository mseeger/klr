function y = klr_mulm1(x,a,nn,nc)
%KLR_MULM1 Internal helper function
%  Y = KLR_MULM3(X,A,NN,NC)
%  Returns Y = M_1(A)'*X for M_1 matrix. Here, PI is given in
%  KLR_INTERN.PI, reduced to the sizes NN, NC.

global klr_intern;

temp=0.5*a./klr_intern.sqpi;
y=temp.*x-klr_mulp(klr_intern.sqpi.*a,nn,nc).*x-temp.* ...
  klr_mulp(klr_intern.pi.*x,nn,nc);
