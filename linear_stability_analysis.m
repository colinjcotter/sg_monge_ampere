% SCRIPT TO FIND THE EIGENVALUES REPRESENTING THE GROWTH RATE OF THE
% INSTABILITY IN THE EADY MODEL FOR THE SEMI-GEOSTROPHIC EQUATIONS
% Ax = wBx
format long
Nsq = 2.5*10^-5;
g = 10;
f = 10^-4;
theta0 = 300;
C = 3*10^-6;
H = 10^4;
k = linspace(0.1,40,100)*10^-6;
N = 100;
w = zeros(1,length(k));
z = zeros(1,N-2);
h = H/(N-1);
for j=1:N-2
    z(j) = j*h;
end
U = C*g*(z - H/2)/f/theta0;

for j=1:length(k)
    %Matrix A
    d = (f*f*theta0*g/h/h + k(j)*k(j)*Nsq*theta0)*U;
    dn1 = -f*f*theta0*g*U(1:N-3)/(h^2) - (C*f*g/h)*ones(1,N-3);
    d1 = (C*f*g/h)*ones(1,N-3) - f*f*theta0*U(2:N-2)*g/(h^2) ;

    A = diag(dn1,-1) + diag(d) + diag(d1,1);

    %Matrix B
    d = (k(j)*Nsq*theta0 + f*f*theta0*g/k(j)/h/h)*ones(1,N-2);
    d1 = (-f*f*theta0*g/k(j)/h/h)*ones(1,N-3);

    B = diag(d1,-1) + diag(d) + diag(d1,1);
    [V,e] = eig(A,B,'vector');
    Eimag = imag(e);
    w(j) = max(Eimag);
end

plot(k,w)
    

