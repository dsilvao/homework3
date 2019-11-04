%% data set
nSamples = 999;
%q-
mu{1} = [0,0]; 
sigma{1} = [1 1; 1 2]; 
%q+
mu{2} = [2,2];
sigma{2} = [2 1; 1 1];
% priors
prior = [0.3; 0.7];
xMin = -5; xMax = 7;
yMin = -4; yMax = 7;
nClass = numel(mu);
[data, classIndex] = generateGaussianSamples(mu, sigma, nSamples, prior);
%LDA
[ind01LDA,ind10LDA,ind00LDA,ind11LDA,pEminLDA] = classifxLDA(data, classIndex, mu, sigma, nSamples, prior);
figure(1); subplot(2,3,6);
plotDecision(data,ind01LDA,ind10LDA,ind00LDA,ind11LDA);
title(sprintf('LDA Pe=%.4f',pEminLDA), 'FontSize', 18);
axis([xMin xMax yMin yMax]); 
% MAP Classification 
[ind01MAP,ind10MAP,ind00MAP,ind11MAP,pEminERM] = classifyMAP(data, classIndex, mu, sigma, nSamples, prior);
figure(2); subplot(2,3,6);
plotDecision(data,ind01MAP,ind10MAP,ind00MAP,ind11MAP);
title(sprintf('MAP Pe=%.4f',pEminERM), 'FontSize', 18);
axis([xMin xMax yMin yMax]);
%% function to generate gaussians
function [data, classIndex] = generateGaussianSamples(mu, sigma, nSamples, prior)

% check if priors adds up to 1
if sum(prior) ~= 1
 error('Priors should add to one!');
end
% sample class indexes 
classTempScalar = rand(nSamples, 1);
priorThresholds = cumsum([0; prior]);
nClass = numel(mu);
data = cell(nClass, 1);
classIndex = cell(nClass, 1);
for idxClass = 1:nClass
 nSamplesClass = nnz(classTempScalar>=priorThresholds(idxClass) & classTempScalar<priorThresholds(idxClass+1));
 % Generating the samples 
 data{idxClass} = mvnrnd(mu{idxClass}, sigma{idxClass}, nSamplesClass);
 % Set class labels
 classIndex{idxClass} = ones(nSamplesClass,1) * idxClass;
end
data = cell2mat(data);
classIndex = cell2mat(classIndex);
end
 
%% LDA Function
function [ind01LDA,ind10LDA,ind00LDA,ind11LDA,pEminLDA] = classifxLDA(data, classIndex, mu, sigma, nSamples, ~)
Sb = (mu{1}'-mu{2}')*(mu{1}'-mu{2}')';
Sw = sigma{1} + sigma{2};
[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); 
xLDA = wLDA'*data'; 
wLDA = sign(mean(xLDA(find(classIndex==2)))-mean(xLDA(find(classIndex==1))))*wLDA;  
%T
bLDA = 0.5*[-(mean(xLDA(find(classIndex==1))).'*sign(mean(xLDA(find(classIndex==1)))) ) + (mean(xLDA(find(classIndex==2))).'*sign(mean(xLDA(find(classIndex==2)))) )  ] ;
discriminantScoreLDA = sign(mean(xLDA(find(classIndex==2)))- mean(xLDA(find(classIndex==1))))*xLDA + bLDA; % flip xLDA accordingly
%ROC Curve
[ROCLDA,tauLDA] = estimateROC(discriminantScoreLDA,classIndex');
probErrorLDA = [ROCLDA(1,:)',1- ROCLDA(2,:)']*[sum(classIndex==1),sum(classIndex==2)]'/nSamples; 
pEminLDA = min(probErrorLDA); 
ind = find(probErrorLDA == pEminLDA);
%smallest error threshold 
decisionLDA = (discriminantScoreLDA >= tauLDA(ind(1))); 
ind00LDA = find(decisionLDA==0 & classIndex'==1); 
ind10LDA = find(decisionLDA==1 & classIndex'==1); 
ind01LDA = find(decisionLDA==0 & classIndex'==2); 
ind11LDA = find(decisionLDA==1 & classIndex'==2); 
end  
function [ROC,tau] = estimateROC(discriminantScoreLDA,label)
% Generate ROC curve samples
Nc = [length(find(label==1)),length(find(label==2))];
sortedScore = sort(discriminantScoreLDA,'ascend');
tau = [sortedScore(1)-1,(sortedScore(2:end)+sortedScore(1:end-1))/2,sortedScore(end)+1];
% thresholds at midpoints of consecutive scores in sorted list
for k = 1:length(tau)
 decision = (discriminantScoreLDA >= tau(k));
 ind10 = find(decision==1 & label==1); p10 = length(ind10)/Nc(1); % probability of false positive
 ind11 = find(decision==1 & label==2); p11 = length(ind11)/Nc(2); % probability of true positive
 ROC(:,k) = [p10;p11];
end 
end 

%% Logistic Linear Classifier 



%% map classifier
function [ind01MAP,ind10MAP,ind00MAP,ind11MAP,pEminERM] = classifyMAP(data, classIndex, mu, sigma, nSamples, prior)

discriminantScoreERM = log(evalGaussian(data',mu{2}',sigma{2}))- log(evalGaussian(data',mu{1}',sigma{1}));
lambdaMAP = [0 1;1 0]; % 0-1 loss values yield MAP decision rule
gammaMAP = (lambdaMAP(2,1)-lambdaMAP(1,1))/(lambdaMAP(1,2)-lambdaMAP(2,2)) * prior(1)/prior(2); % threshold for MAP
decisionMAP = (discriminantScoreERM >= log(gammaMAP));
ind00MAP = find(decisionMAP==0 & classIndex'==1); p00MAP = length(ind00MAP)/sum(classIndex==1); % probability of true negative
ind10MAP = find(decisionMAP==1 & classIndex'==1); p10MAP = length(ind10MAP)/sum(classIndex==1); % probability of false positive
ind01MAP = find(decisionMAP==0 & classIndex'==2); p01MAP = length(ind01MAP)/sum(classIndex==2); % probability of false negative
ind11MAP = find(decisionMAP==1 & classIndex'==2); p11MAP = length(ind11MAP)/sum(classIndex==2); % probability of true positive
pEminERM = [p10MAP,p01MAP]*[sum(classIndex==1),sum(classIndex==2)]'/nSamples; 
% probability of error for MAP classifier. 
end 
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end 

function plotDecision(data,ind01,ind10,ind00,ind11)
plot(data(ind01,1),data(ind01,2),'xm'); hold on; % false negatives
plot(data(ind10,1),data(ind10,2),'om'); hold on; % false positives
plot(data(ind00,1),data(ind00,2),'xg'); hold on;
plot(data(ind11,1),data(ind11,2),'og'); hold on;
xlabel('Feature 1', 'FontSize', 16);
ylabel('Feature 2', 'FontSize', 16);
grid on; box on;
set(gca, 'FontSize', 14);
legend({'Misclassified as C1','Misclassified as C2'}); 
end 

