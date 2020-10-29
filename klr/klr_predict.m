function logpi = klr_predict(alpha,xtest)
%KLR_PREDICT Compute predictive probabilities
%  LOGPI = KLR_PREDICT(ALPHA,XTEST)
%  The predictor is given by the parameters ALPHA, the training
%  input points in KLR_INTERN.XDATA, and the kernel parameters in
%  KLR.COVINFO.THETA. Computes pred. probabilities for the test
%  input points in XTEST (cases are rows). LOGPI contains these
%  probabilities in the rows. KLR.BIAS_PVAR contains the variance
%  on the bias parameter prior.

global klr klr_intern;

nn=klr.num_data; nc=klr.num_class;
if size(alpha,1)~=nn*nc
  error('ALPHA wrong size');
end
nd=size(klr_intern.xdata,2);
if size(xtest,2)~=nd
  error('XTEST wrong size');
end
nte=size(xtest,1);
fst_reshape(alpha,nn,nc);
bias=klr.bias_pvar*sum(alpha,1);
fst_reshape(alpha,nn*nc,1);
ute=klr_wrap_covtestmul(alpha,xtest)+reshape(bias(ones(nte,1),:), ...
					     nte*nc,1);
fst_reshape(ute,nte,nc);
lte=logsumexp(ute')';
logpi=ute-lte(:,ones(nc,1));
