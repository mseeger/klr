function y = klr_divprecond(x,varargin)
%KLR_DIVPRECOND Internal helper for KLR_FINDMAP
%  Y = KLR_DIVPRECOND(X)
%  If BCOND is a preconditioner for a system with matrix B,
%  i.e. BCOND is an approximation to B which can be inverted
%  easily, we return Y = BCOND\X.
%  In the moment, BCOND=DIAG(B), which is given in
%  KLR_INTERN.DIAGMAT (reduced to NEXCIND). The system is reduced
%  to the components listed in KLR_INTERN.NEXCIND.

% Function can be called with >1 argument, all except first are
% ignored. Needed, because additional args are passed to
% KLR_MULSYSMAT in PCG (see KLR_ACCUMCRITGRAD).

global klr_intern;

y=x./klr_intern.diagmat;
