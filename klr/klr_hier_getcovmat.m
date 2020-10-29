function kmat = klr_hier_getcovmat(c,varargin)
%KLR_HIER_GETCOVMAT GETCOVMAT dummy, just throws exception
%  KMAT = KLR_HIER_GETCOVMAT(C,{SUBIND=[]})
%  Function not supported for hierarchical classif.

global klr klr_intern;

if ~isfield(klr,'hierarch') || ~klr.hierarch.use
  error('Hierarchical classification mode not active');
end
error('GETCOVMAT not supported in hierarchical classif. mode');
