

close all;

lambda =    [0.5,   0.2,    0.1,    0.05,   0.01    ]
density =   [7/25,  13/25,  13/25,  22/25,  25/25   ]
accurarcy = [0.593, 0.7,    0.833,  0.933,  0.993   ]

figure;

%plot(lambda,density , '-r')
%line(lambda,accurarcy)

plot(lambda,density,'-ro',lambda,accurarcy,'-xb')
legend('density','accurarcy')
xlabel('lambda')



