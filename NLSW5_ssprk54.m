function [xloc,utrue,ucloc,err_ureal,energy,time] = NLSW5_ssprk54(q,N,T,alpha,tau,beta)

% Solve u_tt - u_xx + iu_t - 2 |u|^2 u = 0 in 1D -50 < x < 50 by DG

% q = degree for u
% q = degree for v for now, can also choose q-1
% N = number of cells
% T is the simulation time
% alpha,tau,beta for the numerical flux

% set up the grid

x = linspace(-50,50,N+1);
h = x(2:N+1) - x(1:N);

% construct the matrices

% get the nodes in reference domain [-1,1] and their weight
[r,W] = GaussQCofs(q+1);
% disp(r);
% disp(W);

% get the q+1*q+1 matrix of legendre polynomials at nodes r; the maximum degree is q 
P = zeros(q+1,q+1);
for d = 1:q+1
    P(:,d) = JacobiP(r,0,0,d-1); 
end 

% get the q+1 * q+1 matrix of the first derivative of legendre polynomials at nodes
% r; the maximum degree is q-1
S = zeros(q+1,q+1);
for d = 1:q+1
    S(:,d) = GradJacobiP(r,0,0,d-1);
end

% express the boundary [-1;1]
bp = [-1; 1];
Vb = zeros(2,q+1); 
Uxb = Vb;
for d = 1:q+1
    Vb(:,d) = JacobiP(bp,0,0,d-1);
    Uxb(:,d) = GradJacobiP(bp,0,0,d-1); 
end 

% Initialize u and v
u_real = zeros(q+1,N);
u_imag = zeros(q+1,N);
v_real = zeros(q+1,N);
v_imag = zeros(q+1,N);


% For initial data compute the L2 projections 
xloc = zeros(q+1,N);
uinit = zeros(q+1,N);
for j=1:N
    xloc(:,j) = (x(j)+x(j+1)+h(j)*r)/2;  
    [uloc, vloc] = solnew(xloc(:,j),0);

    for d = 1:q+1
        u_real(d,j) = (uloc(1,:).*(W'))*P(:,d);
        v_real(d,j) = (vloc(1,:).*(W'))*P(:,d);
    
        u_imag(d,j) = (uloc(2,:).*(W'))*P(:,d); 
        v_imag(d,j) = (vloc(2,:).*(W'))*P(:,d);
    end
    
    uinit(:,j) = (P*u_real(:,j)).^2 + (P*u_imag(:,j)).^2;
    uinit(:,j) = sqrt(uinit(:,j));
end

% % plot the inital projection
% figure
% 
% plot(xloc,uinit,'k');
% xlabel('x');
% ylabel('|u|');
% title('initial projection for |u|');
% 
% % break point 1

% Time stepping - SSPRK(5,4)
CFL = 0.75; 
dt = CFL/(2*pi)*(min(h));
% dt = 0.1*dt;
nsteps = ceil(T/dt);
dt = T/nsteps;

c11 = (0.391752226571890);

c20 = (0.444370493651235);
c21 = (0.555629506348765);
c22 = (0.368410593050371);

c30 = (0.620101851488403);
c32 = (0.379898148511597);
c33 = (0.251891774271694);

c40 = (0.178079954393132);
c43 = (0.821920045606868);
c44 = (0.544974750228521);

c52 = (0.517231671970585);
c53 = (0.096059710526147);
c53_1 = (0.063692468666290);
c54 = (0.386708617503269);
c55 = (0.226007483236906);

% check energy conservation of the time integrator
energy = zeros(nsteps+1,1);
time = zeros(nsteps+1,1);
for j=1:N
    energy(1,1) = (h(j)/2)*((P*v_real(:,j)).*W)'*(P*v_real(:,j));
    energy(1,1) = energy(1,1) + (h(j)/2)*((P*v_imag(:,j)).*W)'*(P*v_imag(:,j));
    
    energy(1,1) = energy(1,1) + (h(j)/2)*(((2/h(j))*S*u_real(:,j)).*W)'*((2/h(j))*S*u_real(:,j));
    energy(1,1) = energy(1,1) + (h(j)/2)*(((2/h(j))*S*u_imag(:,j)).*W)'*((2/h(j))*S*u_imag(:,j));
    
    energy(1,1) = energy(1,1) + (h(j)/2)*((P*u_real(:,j)).*W)'*(P*u_real(:,j));
    energy(1,1) = energy(1,1) + (h(j)/2)*((P*u_imag(:,j)).*W)'*(P*u_imag(:,j));
end

err_ureal = 0;
ucloc = zeros(q+1,N);
ucloc_real = ucloc;
ucloc_imag = ucloc;
vcloc_real = ucloc;
vcloc_imag = ucloc;

utrue = ucloc;
utrue_real = ucloc;
utrue_imag = ucloc;
vtrue_real = ucloc;
vtrue_imag = ucloc;

for it = 1:nsteps
    
    stage = 1;
    
    [rhsu_real, rhsv_real, rhsu_imag, rhsv_imag] = wave_schrodinger_update(dt, it, stage, q, N, h, ...
        S, W, P, alpha, Vb, v_real, Uxb, u_real, tau, beta, v_imag, u_imag);
    u_real1 = u_real + c11*dt*rhsu_real;
    v_real1 = v_real + c11*dt*rhsv_real;    
    u_imag1 = u_imag + c11*dt*rhsu_imag;
    v_imag1 = v_imag + c11*dt*rhsv_imag;
    
    stage = stage + 1;
    
    [rhsu_real, rhsv_real, rhsu_imag, rhsv_imag] = wave_schrodinger_update(dt, it, stage, q, N, h, ...
        S, W, P, alpha, Vb, v_real1, Uxb, u_real1, tau, beta, v_imag1, u_imag1);
    u_real2 = c20*u_real + c21*u_real1 + c22*dt*rhsu_real;
    v_real2 = c20*v_real + c21*v_real1 + c22*dt*rhsv_real;   
    u_imag2 = c20*u_imag + c21*u_imag1 + c22*dt*rhsu_imag;
    v_imag2 = c20*v_imag + c21*v_imag1 + c22*dt*rhsv_imag;
    
    stage = stage + 1;
    
    [rhsu_real, rhsv_real, rhsu_imag, rhsv_imag] = wave_schrodinger_update(dt, it, stage, q, N, h, ...
        S, W, P, alpha, Vb, v_real2, Uxb, u_real2, tau, beta, v_imag2, u_imag2);
    u_real3 = c30*u_real + c32*u_real2 + c33*dt*rhsu_real;
    v_real3 = c30*v_real + c32*v_real2 + c33*dt*rhsv_real;   
    u_imag3 = c30*u_imag + c32*u_imag2 + c33*dt*rhsu_imag;
    v_imag3 = c30*v_imag + c32*v_imag2 + c33*dt*rhsv_imag;
    
    stage = stage + 1;
    
    [rhsu_real3, rhsv_real3, rhsu_imag3, rhsv_imag3] = wave_schrodinger_update(dt, it, stage, q, N, h, ...
        S, W, P, alpha, Vb, v_real3, Uxb, u_real3, tau, beta, v_imag3, u_imag3);
    u_real4 = c40*u_real + c43*u_real3 + c44*dt*rhsu_real3;
    v_real4 = c40*v_real + c43*v_real3 + c44*dt*rhsv_real3;   
    u_imag4 = c40*u_imag + c43*u_imag3 + c44*dt*rhsu_imag3;
    v_imag4 = c40*v_imag + c43*v_imag3 + c44*dt*rhsv_imag3;
    
    stage = stage + 1;
    
    [rhsu_real, rhsv_real, rhsu_imag, rhsv_imag] = wave_schrodinger_update(dt, it, stage, q, N, h, ...
        S, W, P, alpha, Vb, v_real4, Uxb, u_real4, tau, beta, v_imag4, u_imag4);
    u_real = c52*u_real2 + c53*u_real3 + c53_1*dt*rhsu_real3 + c54*u_real4 + c55*dt*rhsu_real;
    v_real = c52*v_real2 + c53*v_real3 + c53_1*dt*rhsv_real3 + c54*v_real4 + c55*dt*rhsv_real;   
    u_imag = c52*u_imag2 + c53*u_imag3 + c53_1*dt*rhsu_imag3 + c54*u_imag4 + c55*dt*rhsu_imag;
    v_imag = c52*v_imag2 + c53*v_imag3 + c53_1*dt*rhsv_imag3 + c54*v_imag4 + c55*dt*rhsv_imag;
    
    for j=1:N
        energy(it+1,1) = (h(j)/2)*((P*v_real(:,j)).*W)'*(P*v_real(:,j));
        energy(it+1,1) = energy(it+1,1) + (h(j)/2)*((P*v_imag(:,j)).*W)'*(P*v_imag(:,j));
    
        energy(it+1,1) = energy(it+1,1) + (h(j)/2)*(((2/h(j))*S*u_real(:,j)).*W)'*((2/h(j))*S*u_real(:,j));
        energy(it+1,1) = energy(it+1,1) + (h(j)/2)*(((2/h(j))*S*u_imag(:,j)).*W)'*((2/h(j))*S*u_imag(:,j));
        
        energy(it+1,1) = energy(it+1,1) + (h(j)/2)*((P*u_real(:,j)).*W)'*(P*u_real(:,j));
        energy(it+1,1) = energy(it+1,1) + (h(j)/2)*((P*u_imag(:,j)).*W)'*(P*u_imag(:,j));
    end
    
    time(it+1,1) = time(it,1) + dt;
    
end

for j=1:N
    [uloc, vloc] = solnew(xloc(:,j),T);
    
    utrue_real(:,j) = (uloc(1,:)');
    utrue_imag(:,j) = (uloc(2,:)');
    utrue(:,j) = (utrue_real(:,j)).^2 + (utrue_imag(:,j)).^2;
    utrue(:,j) = sqrt(utrue(:,j));
    
    vtrue_real(:,j) = (vloc(1,:)');
    vtrue_imag(:,j) = (vloc(2,:)');
  
%   test for u, v, L2 norm
    ucloc_real(:,j) = P*u_real(:,j);
    ucloc_imag(:,j) = P*u_imag(:,j);
    vcloc_real(:,j) = P*v_real(:,j);
    vcloc_imag(:,j) = P*v_imag(:,j);
    
    err_ureal = err_ureal + (h(j)/2)*((ucloc_real(:,j) - utrue_real(:,j)).*W)'*(ucloc_real(:,j) - utrue_real(:,j));

    ucloc(:,j) = (ucloc_real(:,j)).^2 + (ucloc_imag(:,j)).^2;
    ucloc(:,j) = sqrt(ucloc(:,j));
    
end

% plot(xloc(:),ucloc(:),'b',xloc(:),utrue(:),'r--')

fprintf('L2 error = %4.3e \n',sqrt(err_ureal));

end

function [u, v] = solnew(xloc,t)

J = 1/4;
A = abs(J);
theta = - 1/2 - sqrt(3)/4;

u = [(A.*sech(J.*xloc).*cos(theta.*t))';(A.*sech(J.*xloc).*sin(theta.*t))'];
v = [(-theta.*A.*sech(J.*xloc).*sin(theta.*t))';(theta.*A.*sech(J.*xloc).*cos(theta.*t))'];

end

function [u_x] = grad_u(xloc,t)

J = 1/4;
A = abs(J);
theta = - 1/2 - sqrt(3)/4;

u_x = [(-A.*J.*sech(J.*xloc).*tanh(J.*xloc).*cos(theta.*t))';...
    (-A.*J.*sech(J.*xloc).*tanh(J.*xloc).*sin(theta.*t))'];

end

function [ut_real, vt_real, ut_imag, vt_imag] = wave_schrodinger_update(dt, it, stage, ...
    q, N, h, S, W, P, alpha, Vb, v_real, Uxb, u_real, tau, beta, v_imag, u_imag)

c11 = (0.391752226571890);

c20 = (0.444370493651235);
c21 = (0.555629506348765);
c22 = (0.368410593050371);

c30 = (0.620101851488403);
c32 = (0.379898148511597);
c33 = (0.251891774271694);

c40 = (0.178079954393132);
c43 = (0.821920045606868);
c44 = (0.544974750228521);

c52 = (0.517231671970585);
c53 = (0.096059710526147);
c53_1 = (0.063692468666290);
c54 = (0.386708617503269);
c55 = (0.226007483236906);

ut_real = zeros(q+1,N);
vt_real = zeros(q+1,N);
ut_imag = zeros(q+1,N);
vt_imag = zeros(q+1,N);

[r_fine,W_fine] = GaussQCofs(16);
P_fine = zeros(16,q+1);

for d = 1:q+1
    P_fine(:,d) = JacobiP(r_fine,0,0,d-1); 
end

for j = 1:N
            
            % assume u and v of the same order for now
            
            fu2 = (P_fine*u_real(:,j)).^2 + (P_fine*u_imag(:,j)).^2;
            
            Mu = (2/h(j))*(S')*diag(W)*S - 2*(h(j)/2)*(P_fine')*(diag(W_fine)*diag(fu2))*P_fine;           
            Su = Mu;            
            Sv = -Mu;
            Mv = (h(j)/2)*(P')*diag(W)*P;
            

            % Flux
            % BC
            
            if (j==1)
                
                if (stage==1)
                    [u_bdry, v_bdry] = solnew(-50,dt*(it-1) + c11*dt);
                    vstarl_real = (v_bdry(1,:)');
                    vstarl_imag = (v_bdry(2,:)');
                    
                    [u_x] = grad_u(-50,dt*(it-1) + c11*dt);
                    wstarl_real = (u_x(1,:)');
                    wstarl_imag = (u_x(2,:)');
                    
                elseif (stage==2)
                    [u_bdry, v_bdry] = solnew(-50,dt*(it-1) + (c21*c11+c22)*dt);
                    vstarl_real = (v_bdry(1,:)');
                    vstarl_imag = (v_bdry(2,:)');
                    
                    [u_x] = grad_u(-50,dt*(it-1) + (c21*c11+c22)*dt);
                    wstarl_real = (u_x(1,:)');
                    wstarl_imag = (u_x(2,:)');
                    
                elseif (stage==3)
                    coeff = c32*c21*c11 + c32*c22 + c33;
                    [u_bdry, v_bdry] = solnew(-50,dt*(it-1) + coeff*dt);
                    vstarl_real = (v_bdry(1,:)');
                    vstarl_imag = (v_bdry(2,:)');
                    
                    [u_x] = grad_u(-50,dt*(it-1) + coeff*dt);
                    wstarl_real = (u_x(1,:)');
                    wstarl_imag = (u_x(2,:)');
                    
                elseif (stage==4)
                    coeff = c43*c32*c21*c11 + c43*c32*c22 + c43*c33 + c44;
                    [u_bdry, v_bdry] = solnew(-50,dt*(it-1) + coeff*dt);
                    vstarl_real = (v_bdry(1,:)');
                    vstarl_imag = (v_bdry(2,:)');
                    
                    [u_x] = grad_u(-50,dt*(it-1) + coeff*dt);
                    wstarl_real = (u_x(1,:)');
                    wstarl_imag = (u_x(2,:)');
                    
                else
                    % stage == 5
                    [u_bdry, v_bdry] = solnew(-50,dt*it);
                    vstarl_real = (v_bdry(1,:)');
                    vstarl_imag = (v_bdry(2,:)');
                    
                    [u_x] = grad_u(-50,dt*it);
                    wstarl_real = (u_x(1,:)');
                    wstarl_imag = (u_x(2,:)');
                end
                
%                 vstarl_real = (Vb(1,:)*(v_real(:,j)));
                
                vstarr_real = alpha*(Vb(2,:)*(v_real(:,j)));
                vstarr_real = vstarr_real + (1-alpha)*(Vb(1,:)*(v_real(:,j+1)));
                jump = (Uxb(2,:)*(u_real(:,j))) - (Uxb(1,:)*(u_real(:,j+1)));
                vstarr_real = vstarr_real - tau*jump;

                
%                 wstarl_real = (Uxb(1,:)*(u_real(:,j)));
                
                wstarr_real = (1-alpha)*(Uxb(2,:)*(u_real(:,j)));
                wstarr_real = wstarr_real + alpha*(Uxb(1,:)*(u_real(:,j+1)));
                jump = Vb(2,:)*(v_real(:,j)) - Vb(1,:)*(v_real(:,j+1));
                wstarr_real = wstarr_real - beta*jump;
                
                
%                 vstarl_imag = (Vb(1,:)*(v_imag(:,j)));
                
                vstarr_imag = alpha*(Vb(2,:)*(v_imag(:,j)));
                vstarr_imag = vstarr_imag + (1-alpha)*(Vb(1,:)*(v_imag(:,j+1)));
                jump = (Uxb(2,:)*(u_imag(:,j))) - (Uxb(1,:)*(u_imag(:,j+1)));
                vstarr_imag = vstarr_imag - tau*jump;
                
                
%                 wstarl_imag = (Uxb(1,:)*(u_imag(:,j)));
                
                wstarr_imag = (1-alpha)*(Uxb(2,:)*(u_imag(:,j)));
                wstarr_imag = wstarr_imag + alpha*(Uxb(1,:)*(u_imag(:,j+1)));
                jump = Vb(2,:)*(v_imag(:,j)) - Vb(1,:)*(v_imag(:,j+1));
                wstarr_imag = wstarr_imag - beta*jump;
                
            elseif (j==N)
                
                if (stage==1)
                    [u_bdry, v_bdry] = solnew(50,dt*(it-1) + c11*dt);
                    vstarr_real = (v_bdry(1,:)');
                    vstarr_imag = (v_bdry(2,:)');
                    
                    [u_x] = grad_u(50,dt*(it-1) + c11*dt);
                    wstarr_real = (u_x(1,:)');
                    wstarr_imag = (u_x(2,:)');
                    
                elseif (stage==2)
                    [u_bdry, v_bdry] = solnew(50,dt*(it-1) + (c21*c11+c22)*dt);
                    vstarr_real = (v_bdry(1,:)');
                    vstarr_imag = (v_bdry(2,:)');
                    
                    [u_x] = grad_u(50,dt*(it-1) + (c21*c11+c22)*dt);
                    wstarr_real = (u_x(1,:)');
                    wstarr_imag = (u_x(2,:)');
                    
                elseif (stage==3)
                    coeff = c32*c21*c11 + c32*c22 + c33;
                    [u_bdry, v_bdry] = solnew(50,dt*(it-1) + coeff*dt);
                    vstarr_real = (v_bdry(1,:)');
                    vstarr_imag = (v_bdry(2,:)');
                    
                    [u_x] = grad_u(50,dt*(it-1) + coeff*dt);
                    wstarr_real = (u_x(1,:)');
                    wstarr_imag = (u_x(2,:)');
                    
                elseif (stage==4)
                    coeff = c43*c32*c21*c11 + c43*c32*c22 + c43*c33 + c44;
                    [u_bdry, v_bdry] = solnew(50,dt*(it-1) + coeff*dt);
                    vstarr_real = (v_bdry(1,:)');
                    vstarr_imag = (v_bdry(2,:)');
                    
                    [u_x] = grad_u(50,dt*(it-1) + coeff*dt);
                    wstarr_real = (u_x(1,:)');
                    wstarr_imag = (u_x(2,:)');
                    
                else
                    % stage == 5
                    [u_bdry, v_bdry] = solnew(50,dt*it);
                    vstarr_real = (v_bdry(1,:)');
                    vstarr_imag = (v_bdry(2,:)');
                    
                    [u_x] = grad_u(50,dt*it);
                    wstarr_real = (u_x(1,:)');
                    wstarr_imag = (u_x(2,:)');
                end
                
                vstarl_real = vstarr_real;

%                 vstarr_real = (Vb(2,:)*(v_real(:,j)));

                
                wstarl_real = wstarr_real;
                
%                 wstarr_real = (Uxb(2,:)*(u_real(:,j)));

                
                vstarl_imag = vstarr_imag;
                
%                 vstarr_imag = (Vb(2,:)*(v_imag(:,j)));

                
                wstarl_imag = wstarr_imag;
                
%                 wstarr_imag = (Uxb(2,:)*(u_imag(:,j)));

            else
                
                vstarl_real = vstarr_real;

                vstarr_real = alpha*(Vb(2,:)*(v_real(:,j)));
                vstarr_real = vstarr_real + (1-alpha)*(Vb(1,:)*(v_real(:,j+1)));
                jump = (Uxb(2,:)*(u_real(:,j))) - (Uxb(1,:)*(u_real(:,j+1)));
                vstarr_real = vstarr_real - tau*jump;

                                
                wstarl_real = wstarr_real;
                
                wstarr_real = (1-alpha)*(Uxb(2,:)*(u_real(:,j)));
                wstarr_real = wstarr_real + alpha*(Uxb(1,:)*(u_real(:,j+1)));
                jump = Vb(2,:)*(v_real(:,j)) - Vb(1,:)*(v_real(:,j+1));
                wstarr_real = wstarr_real - beta*jump;

                
                vstarl_imag = vstarr_imag;
                
                vstarr_imag = alpha*(Vb(2,:)*(v_imag(:,j)));
                vstarr_imag = vstarr_imag + (1-alpha)*(Vb(1,:)*(v_imag(:,j+1)));
                jump = (Uxb(2,:)*(u_imag(:,j))) - (Uxb(1,:)*(u_imag(:,j+1)));
                vstarr_imag = vstarr_imag - tau*jump;

                
                wstarl_imag = wstarr_imag;
                
                wstarr_imag = (1-alpha)*(Uxb(2,:)*(u_imag(:,j)));
                wstarr_imag = wstarr_imag + alpha*(Uxb(1,:)*(u_imag(:,j+1)));
                jump = Vb(2,:)*(v_imag(:,j)) - Vb(1,:)*(v_imag(:,j+1));
                wstarr_imag = wstarr_imag - beta*jump;
                
            end
            
            Fu_real = (-2/h(j))*(Uxb(2,:).*(Vb(2,:)*(v_real(:,j))) ...
                     - Uxb(1,:).*(Vb(1,:)*(v_real(:,j))));
            Fu_real = Fu_real + (2/h(j))*(Uxb(2,:).*vstarr_real - Uxb(1,:).*vstarl_real);
           
            ut_real(:,j) = -Sv*(v_real(:,j));
            ut_real(:,j) = ut_real(:,j) + (Fu_real');
            ut_real(:,j) = pinv(Mu)*(ut_real(:,j));
            
            
            Fv_real = (Vb(2,:).*wstarr_real - Vb(1,:).*wstarl_real);
            
            vt_real(:,j) = Mv*(v_imag(:,j))-Su*(u_real(:,j));
            vt_real(:,j) = vt_real(:,j) + (Fv_real');
            vt_real(:,j) = Mv\(vt_real(:,j));
            
            
            Fu_imag = (-2/h(j))*(Uxb(2,:).*(Vb(2,:)*(v_imag(:,j))) ...
                     - Uxb(1,:).*(Vb(1,:)*(v_imag(:,j))));
            Fu_imag = Fu_imag + (2/h(j))*(Uxb(2,:).*vstarr_imag - Uxb(1,:).*vstarl_imag);         

            ut_imag(:,j) = -Sv*(v_imag(:,j));
            ut_imag(:,j) = ut_imag(:,j) + (Fu_imag');
            ut_imag(:,j) = pinv(Mu)*(ut_imag(:,j));
            
            
            Fv_imag = (Vb(2,:).*wstarr_imag - Vb(1,:).*wstarl_imag);
            
            vt_imag(:,j) = -Mv*(v_real(:,j))-Su*(u_imag(:,j));
            vt_imag(:,j) = vt_imag(:,j) + (Fv_imag');
            vt_imag(:,j) = Mv\(vt_imag(:,j));
            

end

end