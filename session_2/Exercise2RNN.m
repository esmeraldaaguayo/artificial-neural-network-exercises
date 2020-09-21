% Uses Recurrent neural network approach on time series prediction
% exploring both open and close loop format

%Santa Fe
load("Files/lasertrain.dat");
load("Files/laserpred.dat");

%Standardize data
trainMean = mean(lasertrain);
trainStd = std(lasertrain);
stdTrain = (lasertrain-trainMean)/trainStd;
stdTest = (laserpred - trainMean)/trainStd;

% Create Validation set
trainX1 = stdTrain(1:550);
valX = stdTrain(551:650);
trainX2 = stdTrain(651:1000);
trainX = [trainX1;trainX2];

% try a combination of neurons and lags
lags = [10 20 30 40];
neurons = [30 40 50 60];

%records
bestRMSE =inf;
bestNeuron = 0;
bestLag =0;

for lag=lags
    for neuron=neurons
        %[Y, rmse] = CloseLoop(lag, neuron, trainX, valX);
        [Y, rmse] = OpenLoop(lag, neuron, trainX, valX);
        %[Y, rmse] = CloseLoop(lag, neuron, stdTrain, stdTest);
        %[Y, rmse] = OpenLoop(lag, neuron, stdTrain, stdTest);
        if rmse < bestRMSE
            bestRMSE = rmse;
            bestNeuron = neuron;
            bestLag =lag;
            
        end
        
    end
        
end

%[Y, rmse] = CloseLoop(bestLag, bestNeuron, stdTrain, stdTest);
[Y, rmse] = OpenLoop(bestLag, bestNeuron, stdTrain, stdTest);
disp(rmse);