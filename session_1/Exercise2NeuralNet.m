% Exercise 2 neural network
% AIM: investigate performance of different feedforward neural network
% training algorithms

Xtrain = trainDataset(:,1:2)';
Ttrain = trainDataset(:,3)';

Xval = valDataset(:,1:2)';
Tval = valDataset(:,3)';

Xtest = testDataset(:,1:2)';
Ttest = testDataset(:,3)';

% reformat data
input = con2seq(Xtrain); 
target = con2seq(Ttrain);
valInput = con2seq(Xval);
valTarget = con2seq(Tval);
testInput = con2seq(Xtest);
testTarget = con2seq(Ttest);

% Perform 30 trials
% Keep as record
timeVector = zeros(30,1);
errorValVector = zeros(30,1);
errorTestVector = zeros(30,1);


% for loop
for i=1:30

tic
%net=feedforwardnet(32,'trainlm');
net=feedforwardnet([6,17],'trainlm');
%net=feedforwardnet([2,24,4],'trainlm');
%net=feedforwardnet(33,'trainbfg');
%net=feedforwardnet([16,41],'trainbfg');
%net=feedforwardnet([38,32,12],'trainbfg');
%net=feedforwardnet(27,'traincgf');
%net=feedforwardnet([10,36],'traincgf');
%net=feedforwardnet([10,13,46],'traincgf');

net.trainParam.goal=0.0002;
net = train(net,input,target);
timeElapse = toc;

a=sim(net,valInput);
err = mse(net,a,valTarget);
errorValVector(i) = err;
timeVector(i) = timeElapse;
a2=sim(net,testInput);
errTest = mse(net,a2,testTarget);
errorTestVector(i) = errTest;
end

timeAverage = sum(timeVector)/30;
min = min(timeVector);
max = max(timeVector);
std = std(timeVector);
errorValAverage = sum(errorValVector)/30;
errorTestAverage = sum(errorTestVector)/30;


disp(timeAverage);
%disp(min);
%disp(max);
%disp(std);
disp(errorValAverage);
disp(errorTestAverage);

