% Test hyperparameter optimization

global klr klr_intern;

% Load data
load('satimage.mat');
klr.num_class=6; nc=klr.num_class;
fpath=strcat(getenv('HOME'),'/matlab/logregr/exp/');
nte=size(test,1);
numattr=size(test,2)-1;

% MC separate (code 1)
roc1=zeros(nte,10);
fprintf(1,'MC separate\n');
complete1=(1:10);
for i=complete1
  load(strcat(fpath,'result17-',int2str(i),'.mat'));
  pause;
  roc1(:,i)=roc;
end

% 1rest (code 2)
roc2=zeros(nte,10);
complete2=(1:10);
for i=complete2
  fprintf(1,'1rest, exp=%d\n',i);
  logpi=[];
  for c=1:nc
    load(strcat(fpath,'onerest13-',int2str(i),'-',int2str(c),'.mat'));
    pause;
    lte=logsumexp(reshape(ute,nte,2)')';
    logpite=reshape(ute,nte,2)-lte(:,ones(2,1));
    logpi=[logpi logpite(:,1)];
  end
  [mxval,guess]=max(logpi,[],2);
  [dummy,indte]=sort(-mxval);
  errte=(guess~=test(:,numattr+1));
  roc2(:,i)=cumsum(errte(indte))/nte;
end

% MC tied (code 3)
complete3=(1:10);
roc3=zeros(nte,10);
fprintf(1,'MC tied\n');
for i=complete3
  load(strcat(fpath,'result16-',int2str(i),'.mat'));
  pause;
  roc3(:,i)=roc;
end

compall=intersect(complete1,intersect(complete2,complete3))
aroc1=mean(roc1(:,compall)')';
aroc2=mean(roc2(:,compall)')';
aroc3=mean(roc3(:,compall)')';
%aroc1=mean(roc1(:,complete1)')';
%aroc2=mean(roc2(:,complete2)')';
%aroc3=mean(roc3(:,complete3)')';
x=(1:nte)';
figure(1); clf;
plot(x,aroc1,'-',x,aroc2,'--',x,aroc3,'-.');
comp12=intersect(complete1,complete2);
adiff=mean(roc2(:,comp12)'-roc1(:,comp12)')';
figure(2); clf;
plot(x,adiff,'-');
