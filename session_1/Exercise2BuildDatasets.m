% Exercise 2
% build datasets for exercise 2
 
%Build Tnew
data = load ('Files/Data_Problem1_regression.mat');
Tnew = (9*data.T1 + 6*data.T2 + 6*data.T3 + 3*data.T4 + 3*data.T5)/(9+6+6+3+3);

allData = [data.X1 data.X2 Tnew];

trainDataset = zeros(1000,3);
valDataset = zeros(1000,3);
testDataset = zeros(1000,3);

%training
for element = 1:size(trainDataset,1)
    index = randsample(13600,1);
    trainDataset(element,1) = data.X1(index);
    trainDataset(element,2) = data.X2(index);
    trainDataset(element,3) = Tnew(index);
end

%validation
for element = 1:size(trainDataset,1)
    index = randsample(13600,1);
    valDataset(element,1) = data.X1(index);
    valDataset(element,2) = data.X2(index);
    valDataset(element,3) = Tnew(index);
end

%testing
for element = 1:size(trainDataset,1)
    index = randsample(13600,1);
    testDataset(element,1) = data.X1(index);
    testDataset(element,2) = data.X2(index);
    testDataset(element,3) = Tnew(index);
end

% plot the surface of training dataset to see that the datapoints are
% randomly spaced

x = trainDataset(:,1);
y = trainDataset(:,2);
t = trainDataset(:,3);

% generate an uniformly sample set of data
xlin = linspace(min(x), max(x), 100);
ylin = linspace(min(y), max(y), 100);
[X,Y] = meshgrid(xlin,ylin);

F = scatteredInterpolant(x,y,t);
Z = F(X,Y);

mesh(X,Y,Z);
hold on
plot3(x,y,t,'.', 'MarkerSize',15);

save('datasets', 'trainDataset', 'valDataset', 'testDataset');