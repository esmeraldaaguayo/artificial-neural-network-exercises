%Santa Fe
load("Files/lasertrain.dat");
load("Files/laserpred.dat");

figure;
plot(lasertrain);
xlabel 'Discrete time k';
ylabel 'signal';

figure;
plot(laserpred);
xlabel 'Discrete time k';
ylabel 'signal';