function [TrainData,TrainTarget]=getTimeSeriesTrainData(trainset, p)
% IMPORTANT must create variable with data first use importdata('..dat')
% and then pass the variable to this equation

%clear
%clc
%close all

% p is lag

TrainMatrix=[]; %initializes a matrix
for i=1:p
    % generate p columns from trainset data of 971 size
    % every column advanced one step in time axis
    % last column in TrainMatrix will include data from t=-30 to t=1000
    TrainMatrix=[TrainMatrix,trainset(i:end-p+i)]; 
end


TrainData=TrainMatrix(1:end-1,:)'; %access second to last row and get that and up
TrainTarget=trainset(p+1:end)';
