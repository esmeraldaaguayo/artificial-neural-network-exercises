% Exercise 1
% implement a perceptron model


% Data points
X = [ -0.5 -0.5 +0.3;  ...
      -0.5 +0.5 -0.5];
T = [1 0 1];
plotpv(X,T);

% Train perceptron
net = perceptron;
net = configure(net,X,T);

hold on
linehandle = plotpc(net.IW{1},net.b{1});

counter =0;
E = 1;
while (sse(E)) 
   [net,Y,E] = adapt(net,X,T); 
   linehandle = plotpc(net.IW{1},net.b{1},linehandle);
   drawnow;
   counter = 1+counter;
end
disp(counter); % 35

% predict new point
x = [0.7; 1.2];
y = net(x);

% plot results
plotpv(x,y);
circle = findobj(gca,'type','line');
circle.Color = 'red';

hold on;
plotpv(X,T);
plotpc(net.IW{1},net.b{1});
hold off;