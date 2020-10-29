function y = klr_mulm2(x,a,nn,nc)
%KLR_MULP Internal helper function
%  Y = KLR_MULM2(X,A,NN,NC)
%  Returns Y = M_2(A)'*X for M_2 matrix. Here, PI is given in
%  KLR_INTERN.PI and KLR_INTERN.SQPI (see KLR_MULSYSMAT), reduced
%  to the sizes NN, NC.

global klr_intern;

y=0.5*(a-klr_mulp(klr_intern.pi.*a,nn,nc)).*x./klr_intern.sqpi-a.* ...
  klr_mulp(klr_intern.sqpi.*x,nn,nc);
