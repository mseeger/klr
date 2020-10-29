function y = klr_tilcovpartmul(x,iind,jind)
%KLR_TILCOVPARTMUL Internal helper function
%  Y = KLR_TILCOVPARTMUL(X,IIND,JIND)
%  Returns Y = K*X, where K is kernel matrix plus KLR.BIAS_PVAR
%  times P_data, restricted to indexes IIND, JIND (namely
%  K(IIND,JIND)).

global klr;

nni=length(iind); nc=klr.num_class;
nnj=length(jind);
fst_reshape(x,nnj,nc);
temp=klr.bias_pvar*sum(x,1);
fst_reshape(x,nnj*nc,1);
y=klr_wrap_covpartmul(x,iind,jind)+reshape(temp(ones(nni,1),:),nni* ...
					   nc,1);
