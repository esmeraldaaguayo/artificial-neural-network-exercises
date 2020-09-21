% Long short-term memory (LSTM) networks
% lag of 10

% load data
load("Files/lasertrain.dat");
load("Files/laserpred.dat");

% seperate data into training and testing
dataTrain = lasertrain;
dataTest = laserpred;
data = [dataTrain', dataTest'];
figure
plot(data)
xlabel("Time")
ylabel("Laser measurements")
title("Santa Fe dataset")
hold off

%Partition the training and test data. 
numTimeStepsTrain = floor(0.9*numel(data));

dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);


% standardize data
mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

lag =10;
% Prepare predictors and responses
[XTrain, YTrain] = getTimeSeriesTrainData(dataTrainStandardized',lag);


% Define LSTM Network Architecture
numFeatures = lag;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

% Train LSTM Network
net = trainNetwork(XTrain,YTrain,layers,options);

% Forecast Future Time Steps
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);

net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end-(lag-1):end)');

tempInput = [YTrain(end-(lag-2):end),YPred];
numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,tempInput','ExecutionEnvironment','cpu');
    tempInput = tempInput(2:end);
    tempInput(end+1) = YPred(:,i);
end

%Unstandardize the predictions
YPred = sig*YPred + mu;

YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2))

%Plot the training time series with the forecasted values.
figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
title("Forecast")
legend(["Observed" "Forecast"])

%Compare the forecasted values with the test data.
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
ylabel("Error")
title("RMSE = " + rmse)

% Update Network State with Observed Values
net = resetState(net);
net = predictAndUpdateState(net,XTrain);

YPred = [];

if lag > 1
    minus = lag - 2;
    cushion = YTrain((end-minus):end);
    tempInput2 = [cushion,XTest(:,1)];
end

numTimeStepsTest = numel(XTest);
for i = 1:(numTimeStepsTest)
    if lag == 1
        [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
    elseif lag > 1
        [net,YPred(:,i)] = predictAndUpdateState(net,tempInput2','ExecutionEnvironment','cpu');
        tempInput2 = tempInput2(2:end);
        tempInput2(end+1) = XTest(:,i);
    end
end

YPred = sig*YPred + mu;
rmse = sqrt(mean((YPred-YTest).^2))

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
title("Forecast with Updates")

subplot(2,1,2)
stem(YPred - YTest)
ylabel("Error")
title("RMSE = " + rmse)
