% 4 GMM Components. Different means, covariances and probabilty for each.
% 100 samples
m(:,1) = [1;0]; Sigma(:,:,1) = 0.1*[5 0;0,2]; 
m(:,2) = [-1;0]; Sigma(:,:,2) = 0.1*[3 1;1,3]; 
m(:,3) = [0;1]; Sigma(:,:,3) = 0.1*eye(2);
m(:,4) = [0;-1]; Sigma(:,:,4) = 0.1*[2 0;0,1 ];
classPriors = [0.15,0.30,0.35,0.20]; thr = [0,cumsum(classPriors)];
N = 100; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1),clf, colorList = 'rbgc';
%Data Genration + figure 
for l = 1:4
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    figure(1), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
end
T = array2table(x.'); 
part = cvpartition(T.Var1,'KFold',10);
