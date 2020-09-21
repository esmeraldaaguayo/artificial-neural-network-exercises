% Exercise 1
% Explore Hopfield Networks under different dimensionalities

% Experiments with 2 neurons but dif Q number
%T = [1 1; -1 -1;1 -1]'; % 2x3 array 2 neurons 3 attracted
%T = [1 1; -1 -1;1 -1;-1 1]';
%T = [1 1; -1 -1]';
%T = [1 1;1 -1]';

%T = [ 1 1 1; -1 -1 -1; -1 -1 1]'; % it is forced to stay in 2-D
%2Dim
%T = [1 1]';
%T = [1 1;-1 1]';
T = [1 1;-1 1; 1 -1]';
%T = [1 1;-1 1; 1 -1;-1 -1]';

plot(T(1,:),T(2,:),'r*')%%%%%%%QUESTION ---- PLOT, what is r*?
axis([-1.1 1.1 -1.1 1.1])
title('Hopfield Network State Space')
xlabel('a(1)');
ylabel('a(2)');
hold on;

%axis([-1 1 -1 1 -1 1])
%set(gca,'box','on'); axis manual;  hold on;
%plot3(T(1,:),T(2,:),T(3,:),'r*')
%title('Hopfield Network State Space')
%xlabel('a(1)');
%ylabel('a(2)');
%zlabel('a(3)');
%view([37.5 30]);
%hold on;

net = newhop(T);

[Y,Pf,Af] = sim(net,3,[],T); %Number of Qs
Y

color = 'rgbmy';
for i=1:5
   %a = {rands(3,1)}; %3 dim
   a = {rands(2,1)}; %2 dim
   [y,Pf,Af] = sim(net,{1 20},{},a); 
   record=[cell2mat(a) cell2mat(y)];
   start=cell2mat(a);
   plot(start(1,1),start(2,1),'kx',record(1,:),record(2,:),color(rem(i,5)+1));
   %plot3(start(1,1),start(2,1),start(3,1),'kx', ...
      %record(1,:),record(2,:),record(3,:),color(rem(i,5)+1))
end