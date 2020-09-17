function generateNoisyY
%generation of examples and targets 
x=0:0.05:3*pi;
y=sin(x.^2);
noise= y + 0.2*randn(size(y)); % make a vector of noise of the same dimension randn(size(y)) [multiply by a number to add a little or alot of noise) and add it to the target data y (you are adding two vectors)

save('y', 'noise');

end
