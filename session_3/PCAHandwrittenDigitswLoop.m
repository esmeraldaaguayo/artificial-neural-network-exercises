% load data an image has 256 dimensionality 
load threes -ascii

original = threes;
meanOriginal = mean(threes,2);
zeroMeanData = threes - meanOriginal;

% calculate the covariance of data
coVarMatrix = cov(zeroMeanData); %pxp matrix

% records
errorVector =[];

for i = 1:50
    % extract largest eigenvalues with eigenvectors
    [eigenVectorMatrix, eigenValueMatrix] = eigs(coVarMatrix,i); % pxq matrix 
    
    % get transposed multiplier E
    transE = eigenVectorMatrix';
    
    %reduce dataset
    z = transE*zeroMeanData';

    % reconstruct original image 
    originalHat = (eigenVectorMatrix*z)'+meanOriginal;
    Error = sqrt(mean(mean((original-originalHat).^2)));
    errorVector = [errorVector, Error];
end

% image
figure
plot(errorVector);
xlabel ("Principal Components Number");
ylabel ("Reconstrucion Error");
hold off;
