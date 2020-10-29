function y = klr_mulm3(x,a,nn,nc)
%KLR_MULM3 Internal helper function
%  Y = KLR_MULM3(X,A,NN,NC)
%  Returns Y = M_3(A)'*X for M_3 matrix. Here, PI is given in
%  KLR_INTERN.PI, reduced to the sizes NN, NC.

global klr_intern;

y=a.*(x-klr_mulp(klr_intern.pi.*x,nn,nc))- ...
  klr_mulp(klr_intern.pi.*a,nn,nc).*x;
