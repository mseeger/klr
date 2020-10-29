function y = klr_mulsysmat(x,varargin)
%KLR_MULSYSMAT Internal helper for KLR_FINDMAP
%  Y = KLR_MULSYSMAT(X,{IND})
%  Returns Y = B*X for the system matrix B = I + V'*K*V, reduced to
%  the components listed in KLR_INTERN.NEXCIND. K is the kernel matrix
%  plus KLR.BIAS_PVAR times P_data. Multiplication with kernel matrix
%  is given in KLR.COVMUL. LOG(PI) is given in
%  KLR_INTERN.LOGPI. Also, KLR_INTERN.PI and KLR_INTERN.SQPI
%  contain PI and SQRT(PI). If IND is given and not empty, it is
%  passed to KLR.COVMUL. It is either a subindex or a vector [0;
%  NS], NS subindex size (see COVMUL documentation). In this case,
%  X, Y, and all PI variables must be correspondingly reduced, and
%  KLR_INTERN.NEXCIND must be reduced as well.
%  NOTE: The PI related variables are NOT reduced to
%  KLR_INTERN.NEXCIND. KLR_INTERN.NEXCIND must be sorted in
%  asc. order.

global klr klr_intern;

ind =[];
nn=klr.num_data;
if nargin>1
  ind=varargin{1};
  if ~isempty(ind)
    if ind(1)==0
      if length(ind)~=2
	error('Invalid IND');
      end
      nn=ind(2);
    else
      [nn,b]=size(ind);
      if b~=1
	ind=ind'; nn=b;
      end
    end
  end
end
nc=klr.num_class; n=nn*nc;
if length(klr_intern.nexcind)==n
  % No selection
  dummy=x(:,:); % force copy
  klr_mexmulv(dummy,0,nn,nc,klr_intern.sqpi,klr_intern.pi);
  y=klr_wrap_covmul(dummy,ind);
  klr_addsigsq(y,dummy,klr.bias_pvar,nn,nc);
  klr_mexmulv(y,1,nn,nc,klr_intern.sqpi,klr_intern.pi);
  y=y+x;
else
  dummy=zeros(n,1); dummy(klr_intern.nexcind)=x;
  klr_mexmulv(dummy,0,nn,nc,klr_intern.sqpi,klr_intern.pi);
  dummy2=klr_wrap_covmul(dummy,ind);
  klr_addsigsq(dummy2,dummy,klr.bias_pvar,nn,nc);
  klr_mexmulv(dummy2,1,nn,nc,klr_intern.sqpi,klr_intern.pi);
  y=x+dummy2(klr_intern.nexcind);
end
