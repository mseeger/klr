function [fval,grad] = klr_sparsehyperprior
%KLR_SPARSEHYPERPRIOR Hyperprior pushing for zeros
%  [FVAL,GRAD]=KLR_SPARSEHYPERPRIOR
%  Operates on components KLR.COVINFO.HPRIOR.IND of
%  KLR.COVINFO.THETA. Puts i.i.d. Gamma prior on components. If the
%  component is X, the log density is (A-1)*X - B*EXP(X), where A,
%  B are given in KLR.COVINFO.HPRIOR.PARA, KLR.COVINFO.HPRIOR.PARB.
%  Returns value and gradient of negative log density. GRAD is 0
%  except for positions KLR.COVINFO.HPRIOR.IND.

global klr klr_intern;

comps=klr.covinfo.theta(klr.covinfo.hprior.ind);
fval=(1-klr.covinfo.hprior.para)*sum(comps)+ ...
     klr.covinfo.hprior.parb*sum(exp(comps));
grad=zeros(length(klr.covinfo.theta),1);
grad(klr.covinfo.hprior.ind)=klr.covinfo.hprior.parb*exp(comps)+ ...
    (1-klr.covinfo.hprior.para);
