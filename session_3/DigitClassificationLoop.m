clear all
close all
nntraintool('close');
nnet.guis.closeAllViews();

% Neural networks have weights randomly initialized before training.
% Therefore the results from training are different each time. To avoid
% this behavior, explicitly set the random number generator seed.
%rng('default')


% Load the training data into memory
load('digittrain_dataset.mat');

% Records
preTrainAccVector =[];
fineTuneAccVector =[];
normal1HiddenAccVector =[];
normal2HiddenAccVector =[];

for u = 1:30
 % Layer 1
 hiddenSize1 = 500;
 autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

feat1 = encode(autoenc1,xTrainImages);

% Layer 2
hiddenSize2 = 100;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat2 = encode(autoenc2,feat1);

% Layer 3
hiddenSize3 = 100;
autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat3 = encode(autoenc3,feat2);

% Layer 4
%softnet = trainSoftmaxLayer(feat1,tTrain,'MaxEpochs',400);
softnet = trainSoftmaxLayer(feat3,tTrain,'MaxEpochs',400);
%softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);

% Deep Net
%deepnet = stack(autoenc1,softnet);
%deepnet = stack(autoenc1,autoenc2,softnet);
deepnet = stack(autoenc1,autoenc2,autoenc3,softnet);

% Test deep net
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;
load('digittest_dataset.mat');
xTest = zeros(inputSize,numel(xTestImages));
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end
y = deepnet(xTest);
classAccPreTrained=100*(1-confusion(tTest,y));
preTrainAccVector = [preTrainAccVector,classAccPreTrained];

% Test fine-tuned deep net
xTrain = zeros(inputSize,numel(xTrainImages));
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end
deepnet = train(deepnet,xTrain,tTrain);
y = deepnet(xTest);
classAccFineTuned=100*(1-confusion(tTest,y));
fineTuneAccVector = [fineTuneAccVector,classAccFineTuned];


%Compare with normal neural network (1 hidden layers)
net = patternnet(500);
net=train(net,xTrain,tTrain);
y=net(xTest);
classAccNormal1Hidden=100*(1-confusion(tTest,y));
normal1HiddenAccVector = [normal1HiddenAccVector,classAccNormal1Hidden];

%Compare with normal neural network (2 hidden layers)
net = patternnet([500 100]);
net=train(net,xTrain,tTrain);
y=net(xTest);
classAccNormal2Hidden=100*(1-confusion(tTest,y));
normal2HiddenAccVector = [normal2HiddenAccVector,classAccNormal2Hidden];
disp(u);
end

% Average Accuracies

preTrainAcc = sum(preTrainAccVector)/size(preTrainAccVector,2)
fineTuneAcc = sum(fineTuneAccVector)/size(fineTuneAccVector,2)
normal1HiddenAcc = sum(normal1HiddenAccVector)/size(normal1HiddenAccVector,2)
normal2HiddenAcc = sum(normal2HiddenAccVector)/size(normal2HiddenAccVector,2)






