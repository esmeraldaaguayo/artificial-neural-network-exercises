function RMSD = PCAanalysis(Datafile)
% Load data
data = Datafile'; %pxN matrix

% Preprocess data
dataMean = mean(data);

X = data'; %Nxp matrix
clear data;

% calculate the covariance of data
coVarMatrix = cov(X); % pxp matrix

% extract largest eigenvalues with eigenvectors
[eigenVectorMatrix, eigenValueMatrix] = eigs(coVarMatrix,256); % pxq matrix 
clear coVarMatrix;

% sum of all eigen values but the largest and so on
total = sum(diag(eigenValueMatrix));
specialVector = total - cumsum(diag(eigenValueMatrix));

D = diag(eigenValueMatrix); %qx1 vector
plot(D);

transposedE = eigenVectorMatrix'; % qxp matrix

% calculate the reduced PCA matrix
z = transposedE*X'; %qxp * qxN = qxN matrix
clear transposedE;

% convert reduced PCA matrix to Xhat
F = eigenVectorMatrix; %pxq
tempX = F*z; %pxq * qxN = pxN
clear z;
Xhat = (tempX+dataMean)';
clear dataMean;

% calculate root mean square difference
RMSD = (sqrt(mean(mean((X-Xhat).^2))));
%david.winant@kuleuven.be
end