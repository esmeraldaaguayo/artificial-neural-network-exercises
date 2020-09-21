function [Y, rmse] = OpenLoop(lag, neurons, trainDataset, testDataset)

% Open loop Time Series Prediction

%Generate training datasets with lag number
[X,T] = getTimeSeriesTrainData(trainDataset,lag);

% create feedforward neural net
net = feedforwardnet(neurons,'trainlm');
net.trainParam.epochs=3000;
net = train(net,X,T);

% recurrent network part
Y =[];
input = [X(:,end)]; 
% for loop
for i=1:size(testDataset,1)
    yHat = sim(net,input);
    input = input(2:end);
    input(end+1) = testDataset(i);
    Y = [Y,yHat];
end
rmse = sqrt(mean((testDataset-Y').^2));

figure
plot(Y)
hold on
plot(testDataset,'.-')
hold off
legend(["Prediction" "Test Dataset"])
ylabel("rmse Value="+rmse);
title("Open Loop Prediction lag="+lag+" neurons="+neurons);
end
