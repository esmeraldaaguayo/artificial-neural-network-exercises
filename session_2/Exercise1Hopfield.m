% Exercies1
% Create a Hopfield Network

T = [1 1; -1 -1; 1 -1]'; %2x3

net = newhop(T);

a = {[0 0]'};

[y,Pf,Af] = sim(net,{1 50},{},a);

plot(T(1,:),T(2,:),'g*')
axis([-1.1 1.1 -1.1 1.1])
title('Hopfield Network State Space')
xlabel('a(1)');
ylabel('a(2)');
hold on;
record = [cell2mat(a) cell2mat(y)];
start = cell2mat(a);
hold on
plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:))