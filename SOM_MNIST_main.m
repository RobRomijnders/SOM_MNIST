% Implementation of Self Organizing map for the MNIST dataset
% The code is extended to perform a variant of Nearest Neighbor approach
% in the lattice that is produced by the SOM
% 
% The code is commented and generalized for post on github-pages
% http://robromijnders.github.io/SOM_blog/

tic
clear all
close all
clc

%load MNIST_dataset.mat;
% This code assumes that you have the famous MNIST dataset in your working
% directory.
% Adapt the following line according to your settings
%addpath('C:\Users\User\Dropbox\Dropbox\Machine Learning Projects\ML_summer\matconvnet-master\data\mnist')
addpath('/home/rob/Dropbox/Machine Learning Projects/ML_summer/matconvnet-master/data/mnist');

train_data = loadMNISTImages('train-images-idx3-ubyte');
train_classlabel = loadMNISTLabels('train-labels-idx1-ubyte');
test_data = loadMNISTImages('t10k-images-idx3-ubyte');
test_classlabel = loadMNISTLabels('t10k-labels-idx1-ubyte');

N = size(train_data,2);
Ntest = size(test_data,2);

rw = 28;
cl = 28;

columns = randperm(N,16);
for i = 1:16
    subplot(4,4,i)
    tmp=reshape(train_data(:,columns(i)),rw,cl);
    imshow(double(tmp));
end

% % Select the assigned labels
% trainIdx = find(train_classlabel~=1 & train_classlabel~=8); % find the location of classes 0, 1, 2
% testIdx = find(test_classlabel~=1 & test_classlabel~=8); % find the location of classes 0, 1, 2
trainIdx = 1:N;
testIdx = 1:Ntest;

y_train = [train_classlabel(trainIdx)]';
X_train = [train_data(:,trainIdx)]';

y_test = [test_classlabel(testIdx)]';
X_test = [test_data(:,testIdx)]';


% Note that we use the full 60.000 MNIST dataset and select a subset of
% that
N_select = 1000;  %How many train samples do you want?
N_test_select = 500;    %How many test samples do you want?
N = size(X_train,1);
Ntest = size(X_test,1);

idx = randperm(N,N_select);
X_train = X_train(idx,:);
y_train = y_train(idx);

idx = randperm(Ntest,N_test_select);
X_test = X_test(idx,:);
y_test = y_test(idx);
%
N = size(X_train,1);
Ntest = size(X_test,1);
D = size(X_train,2);

%
lat_size = [10,10];   %Define the lattice size?
ls = lat_size(1)*lat_size(2);
[I,J] = ind2sub(lat_size,1:ls);
lat_ind = [I' J'];
lat_dist = zeros(ls,ls);
for i = 1:ls
    d = bsxfun(@minus,lat_ind,lat_ind(i,:));
    lat_dist(:,i) = sqrt(sum(d.^2,2));
end

%Choose the initial distribution for the neurons
L = unifrnd(0,1,ls,D);
%L = randn(ls,D);


% Phase 1
disp('Start phase 1')
iterations_1 = 1000;
eta_start = 0.1;
sigma_start = norm(lat_size)/2;
tau_2 = iterations_1;
tau_1 = iterations_1/log(sigma_start);

% Collect some weights for visual interpretation
plt = 4;   % In the end, we will track plt^2 weights
collect = zeros(plt^2,iterations_1);
ind_plt = randperm(numel(L),plt^2);
coll_hyp = zeros(2,iterations_1);

for n = 1:iterations_1
    %Get the current eta and sigma
    eta = eta_start*exp(-n/tau_2);
    sigma = sigma_start*exp(-n/tau_1);
    coll_hyp(:,n) = [eta;sigma];
    
    %Randomly pick a sample
    ind = randperm(N,1);
    x = X_train(ind,:);
    
    %Determine the winner
    dist = sum((bsxfun(@minus,L,x)).^2,2);
    [~,I] = min(dist);
    
    % Matrix for x-w_j
    diff = -1*bsxfun(@minus,L,x);
    
    %Get the neighboorhood function with local sigma
    nb = exp(-1*lat_dist(:,I)./sigma);
    upd = bsxfun(@times,diff,nb);
    
    L = L + eta*upd;
    
    collect(:,n) = L(ind_plt);
end

% Phase 2
disp('Start phase 2')
eta_start = 0.01;
iterations_2 = 4000;

sigma_start = sigma;   %continue building on the previous sigma
tau_2 = iterations_2;
%tau_1 = iterations_2/log(sigma_start);
tau_1 = iterations_2;

collect = [collect zeros(plt^2,iterations_2)];
coll_hyp = [coll_hyp zeros(2,iterations_2)];



for n = 1:iterations_2
    %Get the current eta and sigma
    eta = eta_start*exp(-n/tau_2);
    sigma = sigma_start*exp(-n/tau_1);
    coll_hyp(:,iterations_1+n) = [eta;sigma];
    
    %Randomly pick a sample
    ind = randperm(N,1);
    x = X_train(ind,:);
    
    %Determine the winner
    dist = sum((bsxfun(@minus,L,x)).^2,2);
    [~,I] = min(dist);
    
    % Matrix for x-w_j
    diff = -1*bsxfun(@minus,L,x);
    
    %Get the neighboorhood function with local sigma
    nb = exp(-1*lat_dist(:,I)./sigma);
    upd = bsxfun(@times,diff,nb);
    
    L = L + eta*upd;
    
    collect(:,iterations_1 + n) = L(ind_plt);
end

figure
hold on
for i = 1:plt^2
    subplot(plt,plt,i)
    plot(collect(i,:))
    %xlabel('iterations')
    %ylabel('Value of weight')
    [pos_l,pos_w] = ind2sub(size(L),ind_plt(i));
    [pos_l1,pos_l2] = ind2sub(lat_size,pos_l);
    title(sprintf('Trajectory of lattice (%d,%d), weight %d ',pos_l1,pos_l2,pos_w ))
end

%Plot hyperparameters
subplot(plt,plt,i-1)
plot(coll_hyp(1,:))
ylabel('eta')
xlabel('iterations')
title('hyperparameter')
subplot(plt,plt,i)
plot(coll_hyp(2,:))
ylabel('sigma')
xlabel('iterations')
title('hyperparameter')

suptitle(sprintf('Phase 1 iter %d Phase 2 iter %d  \n vertical axis: value of weight ; horizontal axis: iterations',iterations_1 ,iterations_2 ));
hold off

% Visualize the results
disp('collect visualizations')
figure
vis = lat_size.*[rw cl];
vis = zeros(vis);
vis2 = vis;

% For every neuron in the lattice, determine the label
disp('label the lattice')
neighbors = 5;
labels = zeros(ls,1);
for i = 1:ls
    %DO the labelling part
    dist = sum((bsxfun(@minus,X_train,L(i,:))).^2,2);
    %[~,I] = min(dist);
    [~,I] = sort(dist);
    label_near = mode(y_train(I(1:neighbors)));
    %labels(i) = y_train(I);
    labels(i) = label_near;
    
    
    % Do the vis part
    im = reshape(L(i,:),rw,cl);
    im2 = reshape(X_train(I(1),:),rw,cl); 
    [pos_l1,pos_l2] = ind2sub(lat_size,i);
    rw_start = (pos_l1-1)*cl+1;
    cl_start = (pos_l2-1)*rw+1;
    vis(rw_start  :rw_start+rw-1 ,cl_start : cl_start+cl-1 ) = im;
    vis2(rw_start  :rw_start+rw-1 ,cl_start : cl_start+cl-1 ) = im2;
end


labels_t = reshape(labels,lat_size(1),lat_size(2));

imshow(vis,[])
title('target classes 0 1 2 3 4 5 6 7 8 9')
uitable('Data', labels_t, 'Position', [10 10 450 500]);
figure 
imshow(vis2,[])
title('target classes 0 1 2 3 4 5 6 7 8 9')
uitable('Data', labels_t, 'Position', [10 10 450 500]);

% Now classify the testset
disp('Label the testset')
N_test = size(X_test,1);

y_test_pred = zeros(N_test,1);
y_train_pred = zeros(N,1);
%neighbors_test = 10;
for i = 1:N_test
    dist = sum((bsxfun(@minus,L,X_test(i,:))).^2,2);
    [~,I] = min(dist);
    y_test_pred(i) = labels(I);
    %label_near = find((round((exp(-1*lat_dist(:,I)./0.4))*10))>=1);
    %label_near = labels(label_near);
    %disp(label_near)
end
for i = 1:N
    dist = sum((bsxfun(@minus,L,X_train(i,:))).^2,2);
    [~,I] = min(dist);
    y_train_pred(i) = labels(I);
end

acc_test = mean(y_test == y_test_pred');
acc_train = mean(y_train == y_train_pred');
toc