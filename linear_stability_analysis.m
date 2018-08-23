% SCRIPT TO FIND THE EIGENVALUES REPRESENTING THE GROWTH RATE OF THE
% INSTABILITY IN THE EADY MODEL FOR THE SEMI-GEOSTROPHIC EQUATIONS
% Ax = wBx
format long
%set parameters from Cullen [2006]
Nsq = 2.5*10^-5;
g = 10;
f = 10^-4;
theta0 = 300;
C = 3*10^-6;
H = 10^4;

%introduce array of wavenumbers, k
k = linspace(0.01,5,500)*10^-6;
N = 100;

%initialise array store omega values 
w = zeros(1,length(k));

%set up discretised values for base velocity
z = zeros(1,N-2);
h = H/(N-1);
for j=1:N-2
    z(j) = j*h;
end
U = C*g*(z - H/2)/f/theta0;

%Solve eigenvalue problem for each value of wavenumber k
for j=1:length(k)
    %Initialise Matrix A
    d = (2*(f^2)*theta0*k(j)/h/h + (k(j)^3)*Nsq*theta0)*U;
    dn1 = -f*f*theta0*k(j)*U(2:N-2)/(h^2) - (C*f*g*k(j)/h)*ones(1,N-3);
    d1 = (C*f*g*k(j)/h)*ones(1,N-3) - f*f*theta0*k(j)*U(1:N-3)/(h^2) ;

    A = diag(dn1,-1) + diag(d) + diag(d1,1);

    %Initialise Matrix B
    d = ((k(j)^2)*Nsq*theta0 + 2*f*f*theta0/h/h)*ones(1,N-2);
    d1 = (-f*f*theta0/h/h)*ones(1,N-3);

    B = diag(d1,-1) + diag(d) + diag(d1,1);
    
    %solve eigenvalue problem
    [V,e] = eig(A,B,'vector');
    Eimag = imag(e);
    
    %extract fastest growing eigenvalue
    w(j) = max(Eimag);
end

plot(k,w)
title('Linear Stability Analysis for Semi-geostrophic Equations','Interpreter','latex')
xlabel('Wavenumber, $k$','Interpreter','latex')
ylabel('Growth Rate, $\omega$','Interpreter','latex')

