function RMSD = PCAanalysisWithForLoop(Datafile)
% for loop exercise
xAxis = 1:50;
yAxis =[];

% Load data
data = Datafile'; %pxN matrix

% Preprocess data
dataMean = mean(data);

X = data'; %Nxp matrix


% calculate the covariance of data
coVarMatrix = cov(X); % pxp matrix


for i=1:50
    [eigenVectorMatrix, eigenValueMatrix] = eigs(coVarMatrix,i);
    transposedE = eigenVectorMatrix'; % qxp matrix

    % calculate the reduced PCA matrix
    z = transposedE*X'; %qxp * qxN = qxN matrix

    % convert reduced PCA matrix to Xhat
    F = eigenVectorMatrix; %pxq
    tempX = F*z; %pxq * qxN = pxN
    Xhat = (tempX+dataMean)';

    % calculate root mean square difference
    %RMSD = (sqrt(mean(mean((X-Xhat).^2))));
    yAxis(i) = (sqrt(mean(mean((X-Xhat).^2))));
    
end

%plot errors
plot(xAxis, yAxis);
RMSD = 0;
end