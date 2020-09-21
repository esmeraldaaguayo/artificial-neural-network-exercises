function [Y, rmse] = CloseLoop(lag, neurons, trainDataset, testDataset)
% Close Loop Time Series Prediction

%Generate training datasets with lag number
[X,T] = getTimeSeriesTrainData(trainDataset,lag);

% create feedforward neural net
net = feedforwardnet(neurons,'trainlm');
net.layers{1}.transferFcn = 'tansig';
net.trainParam.epochs=3000;
net = train(net,X,T);

% recurrent network part
Y =[];
input = X(:,end);   % column
% for loop
for i=1:(size(testDataset,1)+1)
    yHat = sim(net,input);
    input = input(2:end);
    input(end+1) = yHat;
    Y = [Y,yHat];
end

rmse = sqrt(mean((testDataset-Y(2:end)').^2));

figure
plot(Y)
hold on
plot(testDataset,'.-')
hold off
legend(["Prediction" "Test Dataset"])
ylabel("rmse Value="+rmse);
title("Close Loop Prediction lag="+lag+" neurons="+neurons);
end


