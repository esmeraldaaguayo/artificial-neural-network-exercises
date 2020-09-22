% load data an image has 256 dimensionality 
load threes -ascii

original = threes;
meanOriginal = mean(threes,2);
% question = original(2,:);
meanThree = mean(threes,1);
zeroMeanData = threes - meanOriginal;

% calculate the covariance of data
coVarMatrix = cov(zeroMeanData); %pxp matrix

% extract largest eigenvalues with eigenvectors
[eigenVectorMatrix, eigenValueMatrix] = eigs(coVarMatrix,256); % pxq matrix 

% sum of all eigen values but the largest and so on
diagVector=diag(eigenValueMatrix);
cumVector50 = cumsum(diagVector);
first50 = cumVector50(50);
cumVectorRest = cumsum(diagVector(51:end));
last = cumVectorRest(end);

% plot eigenValues
figure
diagonalValues = diag(eigenValueMatrix); %qx1 vector
plot(diagonalValues);
hold off;

% get transposed multiplier E
transE = eigenVectorMatrix';

%reduce dataset
z = transE*zeroMeanData';

% reconstruct original image 
originalHat = (eigenVectorMatrix*z)'+meanOriginal;
Error = sqrt(mean(mean((original-originalHat).^2)));
disp(Error);

% image
figure
colormap('gray');
imagesc(reshape(original(2,:),16,16),[0,1]);
%title original;
hold off;

figure
colormap('gray');
imagesc(reshape(originalHat(2,:),16,16),[0,1]);
%title reconstructed;
hold off;

figure
colormap('gray');
imagesc(reshape(meanThree,16,16),[0,1]);
%title original;
hold off;




