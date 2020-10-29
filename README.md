# klr
Matlab code for efficient multiple kernel logistic regression

## Publication

Details behind the implementation are given in [M. Seeger: Cross-Validation Optimization for Large Scale Structured Classification Kernel Methods](http://www.jmlr.org/papers/volume9/seeger08b/seeger08b.pdf). If you use this code for scientific work, please cite this paper and provide a link to the code.

## Installation

- Make sure to first install the [essential](https://github.com/mseeger/essential) package.
- Compile MEX files, by running make. This produces DLL files.
- Copy all DLL and *.m files somewhere into your Matlab path.

## How to use it

The mathematical details of the implementation are described in the paper linked above. First of all, there are two different things you can do:
- A) Fit the predictor for fixed hyperparameters (kernel parameters) which have been chosen otherwise
- B) Learn the hyperparameters from the data, together with fitting the predictor

B) requires a sequence of calls to A). It is simpler to adapt the code to your particular situation for A) only.

In any case, the global structure KLR is used to provide parameters and handles of primitives to the code. The global structure KLR_INTERN is used for communicating internal variables.

A) is dealt with by the function KLR_FINDMAP, which finds the parameters ALPHA for fixed hyperparameters. Prediction for ALPHA and hyperparameters is done with KLR_PREDICT.

B) is dealth with by the function KLR_CRITFUNC, which evaluates learning criterion and gradient at a given hyoerparameter vector. You can plug this function into your favourite nonlinear optimizer. A simple, free conjugate gradients optimizer is MINIMIZE provided by Carl Rasmussen. Both functions need to be configured by KLR and KLR_INTERN fields, described below.

Both A) and B) need to solve linear systems with the Hessian. This is done approximately by default, using Matlab LCG. If KLR.DEBUG.SOLVEEXACT==1, the systems are solved exactly. This may take more memory and significantly more time. However, it is recommended for small dataset sizes, and in order to debug new code. Some primitives need to be provided only for this exact mode.


## KLR fields and primitives

Here, we describe what KLR, KLR_INTERN fields have to be set, and what primitives (related to the kernel function) have to be provided for the settings A) and B) to work. Note that B) requires A) in any case.

### Support for scenario A) (no hyperparameter learning)

General parameters:
- KLR.NUM_DATA: Number of training datapoints
- KLR.NUM_CLASS: Number of classes
- KLR.VERBOSE: Verbosity level (0: no output; 1: some output; 2: a lot of output)
- KLR.BIAS_PVAR: Variance sigma^2 of the prior on the bias parameters b, note that b is in fact eliminated, see [1]. The smaller this parameter, the more we expect balanced classes. ATTENTION: A large value can result in numerical problems. The value 16 worked fine for us so far.
- KLR_INTERN.XDATA: Training input point matrix, cases are rows. Required by the primitives to recompute kernel matrices when needed

Parameters related to kernel functions:
The structure KLR.COVINFO contains fields describing the kernel configuration.
- KLR.COVINFO.THETA: Hyperparameter vector. Exactly this vector is
  optimized over in B) Coefficients must be unconstrained real. Kernel
  matrix evaluation must be possible given this vector and the input
  points (in KLR_INTERN.XDATA).
  Can contain more than just the kernel parameters. Kernel evaluation
  code assumes that the part for the kernel parameters is contiguous
  starting at position KLR.COVINFO.THPOS (def.: 1). Semantics and size
  of vector depends on kernel configuration.
- KLR.COVINFO.TIED: Boolean flag. If ==1, all kernels are the same,
  otherwise each class has a different kernel. In the latter case, kernels
  may still share parameters, details depend on kernel configuration.

Parameters and handles for the kernel configuration:
A kernel configuration is given by a number of primitives and has to be
supplied by the user, some standard situations are provided. The
primitives maintain some inner kernel matrix representation which has to
be updated whenever kernels pars (KLR.COVINFO.THETA) or training input
points are changed. A fixed part of the repres. is KLR.COVDIAG, apart
from that it should be kept in KLR_INTERN.
- KLR.COVINFO.PREC_OK: Boolean flag. ==1 iff the kernel matrix repres.
  is up-to-date. Code which changes KLR.COVINFO.THETA or KLR_INTERN.XDATA,
  has to set this to 0. Calling the COMP_PREC primitive updates the
  repres. and sets the flag to 1.
- KLR.COVDIAG: Part of the kernel matrix repres., stores kernel matrix
  diagonals stacked on top of each other.
  NOTE: Even if KLR.COVINFO.TIED==1, the diag. for each class kernel is
  stored.
- KLR.COMP_PREC: Handle to COMP_PREC primitive (below)
- KLR.COVMUL: Handle to COVMUL primitive (below)
- KLR.COVPARTMUL: Handle COVPARTMUL primitive (below)
- KLR.COVTESTMUL: Handle to COVTESTMUL primitive (below)
- KLR.COVMULMAT: Handle to COVMULMAT primitive (below)
- KLR.GETCOVMAT: Handle to GETCOVMAT primitive (below). Only required for
  exact solution mode

Parameters for outer loop Newton-Raphson optimization:
- KLR.TOL: Stops if rel. improvement falls below this
- KLR.MAXITER: Stops after this many iterations
- KLR.MAXITER_FB: Optional. Alternative value of KLR.MAXITER used in
  situations where the starting value for ALPHA is not well chosen.
  Should be larger than KLR.MAXITER
- KLR.LPITHRES: Controls stability measure in KLR_FINDMAP, which treat
  components in PI equal to 0 if they are smaller than
  EXP(KLR.LPITHRES).

Parameters for inner loop LCG (solve linear systems):
These are not required in exact solution mode.
- KLR.CG_TOL: Tolerance parameter for LCG
- KLR.CG_MAXIT: Max. number of iterations in LCG
  NOTE: This parameter has to be chosen carefully, esp. in setting B).
  Too small means significant errors in the learning criterion (in
  setting B))

Primitives for the kernel configuration:
These need to be implemented by the user for the specific kernel and
specific kernel matrix representation of the application. For the case
that kernel matrices can be stored in memory, generic implementations
are given in KLR_GEN_XXXXX and some MEX files (below). For some standard
kernels (and storing matrices in memory), all primitive implementations
are provided.
- COMP_PREC:
  COMP_PREC({NEWDATA=0})
  Computes kernel matrix repres. for new kernel parameters THETA, starting
  from pos. KLR.COVINFO.THPOS (def.: 1) in KLR.COVINFO.THETA. This
  is done indep. of the value of KLR.COVINFO.PREC_OK, and this flag is set
  to 1. The kernel matrix diag. is also computed and stored in KLR.COVDIAG.
  The representation depends on the data matrix (provided in
  KLR_INTERN.XDATA) and KLR.COVINFO.THETA. A part of the repres. may depend
  on the input points only (example: the matrix of all squared distances for
  a isotropic kernel). If NEWDATA==1, the complete repres. has to be
  recomputed. If NEWDATA==0, the input points have not been changed since
  the last call of COMP_PREC, but THETA has.
  ATTENTION: COMP_PREC does the recomp. whatever the value of
  KLR.COVINFO.PREC_OK. All other primitives which require the kernel matrix
  repres., have to call COMP_PREC before anything else if
  KLR.COVINFO.PREC_OK==0!
- COVMUL:
  Y = COVMUL(X,{IND=[]})
  Implements Y = K*X (if IND==[]), or Y = K(SI,SI)*X, SI a subset index.
  If IND(1)==0, IND must be [0; NS], and SI==1:NS. Otherwise, SI==IND. SI
  must be increasing.
  The first case can be used for a more efficient implementation, together
  with COVSHUFFLE (below). It is used only in setting B)
  Here, SI selects a subset of the data, therefore K(SI,SI) is short for the
  block-diagonal matrix with blocks K^(c)(SI,SI).
  NOTE: COVMUL with general IND is much less efficient than without IND or
  with IND==[0; NS].
- COVPARTMUL:
  Y = COVPARTMUL(X,IIND,JIND)
  Implements Y = K(IIND,JIND)*X. Here, IIND and JIND are indexes selecting
  subsets of the data, therefore K(IIND,JIND) is short for the block-diagonal
  matrix with blocks K^(c)(IIND,JIND).
  NOTE: COVPARTMUL(X,IND,IND) is much less efficient than COVMUL(X,IND)!
- COVTESTMUL:
  Y = COVTESTMUL(X,TEDATA)
  Implements Y = K*X, where K is the kernel matrix between the test input
  points TEDATA and the training points in KLR_INTERN.XDATA. The kernel
  parameters are in KLR.COVINFO.THETA. X is a vector.
  Does not use the kernel matrix representation.
- COVMULMAT:
  Y = COVMULMAT(X)
  Same as COVMUL, but X, Y are matrices. Subindexes cannot be used.
- GETCOVMAT:
  K = GETCOVMAT(C,{SUBIND=[]})
  Returns kernel matrix for class C. If SUBIND is given and not empty,
  the corr. subpart of the kernel matrix is returned. SUBIND must be
  increasing.
  NOTE: Used only in exact solution mode.

### Support for scenario B) (hyperparameter learning)

All support of 3.1) is required in setting B) as well.

Additional parameters:
- KLR.CVCG_TOL: Optional. Replaces KLR.CG_TOL in the LCG runs used for
  criterion and gradient comp., but not in the KLR_FINDMAP calls
- KLR.CVCG_MAXIT: Optional. Replaces KLR.CG_MAXIT in the LCG runs used for
  criterion and gradient comp., but not in the KLR_FINDMAP calls
- KLR.CVCRIT.IIND, KLR.CVCRIT.JIND: Cell arrays of the same size NFOLD.
  Describes partition to be used for CV criterion. For a fold F, IIND{F}
  is the left-out index I, JIND{F} its complement J. All entries must be
  increasing indexes.
- KLR.ACCUM_GRAD: Handle for ACCUM_GRAD primitive
- KLR.COVSHUFFLE: Handle for COVSHUFFLE primitive

Additional primitives for kernel configuration:
- ACCUM_GRAD:
  GRAD = ACCUM_GRAD(EMAT,FMAT)
  Let NN, NC be number of data, classes. EMAT, FMAT are matrices of size
  (NN*NC)-by-NQ. Computes gradient, where GRAD(i) is the trace of
  FMAT'*((d K)/(d THETA(i)))*EMAT.
  The function makes use of K being block-diagonal. THETA is the kernel
  parameter vector as used by COMP_PREC, etc., which is a part of
  KLR.COVINFO.THETA.
- COVSHUFFLE (handle in KLR.COVSHUFFLE):
  COVSHUFFLE(IND,CIND,FLAG)
  Used together with special 1:NS IND argument to COVMUL. IND and CIND
  must form a partition of 1:KLR.NUMDATA, both increasing. If FLAG==1,
  the kernel matrices are shuffled s.t. rows/cols corr. to IND become 1:NS,
  NS size of IND. If FLAG==0, the inverse shuffling is done. For
  implementations not storing kernel matrices explicitly, the equivalent
  has to be done on the kernel matrix representation.
  Code will call COVMUL with 1:NS IND argument several times within a frame
  COVSHUFFLE(IND,1) ... COVSHUFFLE(IND,0), while no other calls to COVMUL or
  other KLR.XXX functions are done in such a frame. See KLR_ACCUMCRITGRAD
  for an example.
  NOTE: KLR.COVDIAG has to be shuffled accordingly here!
  NOTE: After shuffling, i.e. COVSHUFFLE(IND,1), only the upper left block
  of size NS-by-NS of the kernel matrix repres. must be correct, because
  only this part is used before inverse shuffling. For example, we can store
  two kernel matrices in the upper and lower triangle of a single A. If
  fi=[IND; CIND], COVSHUFFLE(IND,1) can be done as A=A(fi,fi), and
  COVSHUFFLE(IND,0) afterwards as A(fi,fi)=A. For the shuffled A, only the
  part A(1:NS,1:NS) is correct, in the remaining matrix the orig. upper and
  lower triangles are mixed up.

Contiguous subindex feature for COVMUL (COVSHUFFLE):
Relevant for parameter learning. Many systems with a fixed submatrix
have to be solved, which requires calling COVMUL many times with the same
index IND. COVMUL can be implemented more efficiently for IND==1:NS.
To support this, the code allows for blocks
  COVSHUFFLE(IND,CIND,1);
  ...
  COVSHUFFLE(IND,CIND,0);
CIND is the complement of IND, both must be increasing. COVSHUFFLE(IND,CIND,1)
permutes the repres. of the kernel matrices s.t. pos. IND become pos. 1:NS,
NS size of IND. COVSHUFFLE(IND,CIND,0) undoes the shuffling. Within such
blocks, the only KLR.XXX function called is COVMUL(X,[0; NS]), where
[0; NS] encodes the special index 1:NS, being IND after shuffling.

### Generic implementation of primitives

A generic implementation is given for the case where the kernel matrices
are stored explicitly in KLR_INTERN.COVMAT. The primitives are KLR_GEN_XXX.
This excludes the primitives COMP_PREC, COVTESTMUL, and ACCUM_GRAD, which
cannot be done generically.
If KLR.COVINFO.TIED==1, the single kernel matrix is in KLR_INTERN.COVMAT.
Otherwise, KLR_INTERN.COVMAT is a cell array of size CEIL(NC/2), containing
the kernel matrices in successive lower and upper triangles: K^(1) in
lower of COVMAT{1}, K^(2) in upper of COVMAT{1}, ...

The following kernels are imolemented in the moment:
- Gaussian (RBF) kernel: See KLR_RBFCF_COMP_PREC. Generic primitives
- Squared-exponential kernel: See KLR_SQEXPCF_COMP_PREC. Generic primitives
- Linear kernel: See (3.4)

### Primitives for the linear kernel

These are given in KLR_LINCF_XXX. No kernel matrices need to be stored, but
the data matrix (NN rows) must be given in KLR_INTERN.XDATA. This can be
a sparse matrix (and should be, if your features are sparse). KLR.COVINFO.TIED
is ignored, we always allow for different hyperpars for each class:
  K^(c)(x,y) = v_c x^T y + s_c, v_c>0, s_c>=0.
The log(v_c) parameters must be in KLR.COVINFO.THETA at positions
KLR.COVINFO.VPAR_POS. The s_c parameters are used iff KLR.COVINFO.SPAR_POS
is not empty, in this case it contains the pos. of the log(s_c) in
KLR.COVINFO.THETA.
NOTE: In case of the linear kernel, KLR.COVINFO.THPOS is ignored, and the
gradient returned by ACCUM_GRAD has the same size as KLR.COVINFO.THETA;


## Extensions

### Wrappers of kernel primitives

Some extensions of the basic model can be implemented by simply replacing
kernel matrices by simple mappings of the true underlying kernel matrices.
An example is the use of a mixing matrix B described in [1]. To support
this, wrappers KLR_WRAP_XXXXX are defined for the primitives COVMUL,
COVPARTMUL, COVTESTMUL, and ACCUM_GRAD. The generic code calls these
instead of the primitives, i.e. KLR_WRAP_XXXXX(...) instead of
FEVAL(KLR.XXXXX,...). Of course, the wrappers require the primitives just
as well.
By default, the wrappers just call the underlying primitives, but they
can be configured to do other things (see code).
