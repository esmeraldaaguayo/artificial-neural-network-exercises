% Exercise2 Error levels
% AIM: generate the error levels curve for trainlm training algorithm
Xtrain = trainDataset(:,1:2)';
Ttrain = trainDataset(:,3)';

Xval = valDataset(:,1:2)';
Tval = valDataset(:,3)';

Xtest = testDataset(:,1:2)';
Ttest = testDataset(:,3)';

% better format
input = con2seq(Xtrain); 
target = con2seq(Ttrain);
valInput = con2seq(Xval);
valTarget = con2seq(Tval);
testInput = con2seq(Xtest);
testTarget = con2seq(Ttest);

% Build neural network
net=feedforwardnet([6,17],'trainlm');
net = train(net,input,target);
a=sim(net,valInput);
a2=sim(net,testInput);

a21 = cell2mat(testTarget)';
a22 = cell2mat(a2)';

err = a21-a22;

% plot the surface of training dataset to see that the datapoints are
% randomly spaced

x = testDataset(:,1);
y = testDataset(:,2);
t = testDataset(:,3);

% generate an uniformly sample set of data'
xlin = linspace(min(x), max(x), 100);
ylin = linspace(min(y), max(y), 100);
[X,Y] = meshgrid(xlin,ylin);

F = scatteredInterpolant(x,y,err);
Z = F(X,Y);

mesh(X,Y,Z);
hold on
plot3(x,y,tp,'.', 'MarkerSize',15);
