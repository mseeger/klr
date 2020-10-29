function y = klr_mulw(x,nn,nc)
%KLR_MULW Internal helper function
%  Y = KLR_MULW(X,NN,NC)
%  Returns Y = W*X. The size is given by NN, NC. PI is given in
%  KLR_INTERN.PI (reduced to size NN,NC).

global klr_intern;

y=klr_intern.pi.*x;
y=y-klr_intern.pi.*klr_mulp(y,nn,nc);
