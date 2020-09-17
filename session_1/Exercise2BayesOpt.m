% Exercise 2 neural network

Xtrain = trainDataset(:,1:2)';
Ttrain = trainDataset(:,3)';

Xval = valDataset(:,1:2)';
Tval = valDataset(:,3)';

% better format
input = con2seq(Xtrain); 
target = con2seq(Ttrain);
valInput = con2seq(Xval);
valTarget = con2seq(Tval);

% hyperparameter tuning

% Define hyperparameters to optimize
i = optimizableVariable('i',[1,50], 'Type','integer');
j = optimizableVariable('j',[1,50], 'Type','integer');
k = optimizableVariable('k',[1,50], 'Type','integer');
vars = [i,j, k];
%optimize
fun = @(B)myfeedforwardnet(input, target, B.i, B.j, B.k, valInput, valTarget);
result = bayesopt(fun, vars);
[best, criteria] = bestPoint(result);
plot(result,@plotObjectiveModel);

function perf = myfeedforwardnet(x,y,i, j, k, valInput,valTarget)
% create feedforward neural net
net=feedforwardnet([i,j,k], 'traincgf');
net = train(net,x,y);
a=sim(net,valInput);
perf = perform(net,a,valTarget);
end