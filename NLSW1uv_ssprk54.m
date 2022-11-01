function [xloc_u,utrue_real,ucloc_real,err_ureal,err_uimag,err_vreal,err_vimag,energy,time] = NLSW1uv_ssprk54(q,N,T,alpha,tau,beta)

% Solve u_tt - u_xx + iu_t + u = 0 in 1D 0 < x < 2pi by DG
% but the inital condition is not 0, thus
% Solve u_tt - u_xx + iu_t + u + 2e^{ix} = 0 in 1D 0 < x < 2pi by DG
% periodic boundary condition

% q = degree for u
% q-1 = degree for v
% N = number of cells
% T is the simulation time
% alpha,tau,beta for the numerical flux

% set up the grid

x = linspace(0,2*pi,N+1);
h = x(2:N+1) - x(1:N);

% construct the matrices

% get the nodes in reference domain [-1,1] and their weight
[ru,Wu] = GaussQCofs(q+1);
[rv,Wv] = GaussQCofs(q);
% disp(r);
% disp(W);

% get the q+1*q+1 matrix of legendre polynomials at nodes r; the maximum degree is q 
Pu = zeros(q+1,q+1);
for d = 1:q+1
    Pu(:,d) = JacobiP(ru,0,0,d-1); 
end

Pv = zeros(q,q);
Pv_M = zeros(q+1,q);
for d = 1:q
    Pv(:,d) = JacobiP(rv,0,0,d-1);
    Pv_M(:,d) = JacobiP(ru,0,0,d-1);
end

% get the q+1 * q+1 matrix of the first derivative of legendre polynomials at nodes
% r; the maximum degree is q-1
S_u = zeros(q+1,q+1);
for d = 1:q+1
    S_u(:,d) = GradJacobiP(ru,0,0,d-1);
end

S_v = zeros(q,q);
Sv_M = zeros(q+1,q);
for d = 1:q
    S_v(:,d) = GradJacobiP(rv,0,0,d-1);
    Sv_M(:,d) = GradJacobiP(ru,0,0,d-1);
end

% express the boundary [-1;1]
bp = [-1; 1];
Vb_u = zeros(2,q+1); 
Uxb_u = Vb_u;
for d = 1:q+1
    Vb_u(:,d) = JacobiP(bp,0,0,d-1);
    Uxb_u(:,d) = GradJacobiP(bp,0,0,d-1); 
end

Vb_v = zeros(2,q); 
Uxb_v = Vb_v;
for d = 1:q
    Vb_v(:,d) = JacobiP(bp,0,0,d-1);
    Uxb_v(:,d) = GradJacobiP(bp,0,0,d-1); 
end

% Initialize u and v
u_real = zeros(q+1,N);
u_imag = zeros(q+1,N);
v_real = zeros(q,N);
v_imag = zeros(q,N);


% For initial data compute the L2 projections 
xloc_u = zeros(q+1,N);
xloc_v = zeros(q,N);
for j=1:N
    xloc_u(:,j) = (x(j)+x(j+1)+h(j)*ru)/2;
    
    xloc_v(:,j) = (x(j)+x(j+1)+h(j)*rv)/2;
    
    [uloc_u, grad_uloc_u, vloc_u] = solnew(xloc_u(:,j),0);
    [uloc_v, grad_uloc_v, vloc_v] = solnew(xloc_v(:,j),0);

    for d = 1:q+1
        u_real(d,j) = (uloc_u(1,:).*(Wu'))*Pu(:,d);
        u_imag(d,j) = (uloc_u(2,:).*(Wu'))*Pu(:,d);
    end
    
    for d = 1:q
        v_real(d,j) = (vloc_v(1,:).*(Wv'))*Pv(:,d);
        v_imag(d,j) = (vloc_v(2,:).*(Wv'))*Pv(:,d);
    end
end

% % plot the inital projection
% ucloc_real = Pu*u_real;
% ucloc_imag = Pu*u_imag;
% vcloc_real = Pv*v_real;
% vcloc_imag = Pv*v_imag;
% figure

% % top plot
% subplot(2,1,1);
% plot(xloc_u(:),ucloc_real(:),'r--',xloc_v(:),vcloc_real(:),'b--');
% legend({'u','v'},'Location','southeast')
% title('initial projection for the real part')
% 
% % bottom plot
% subplot(2,1,2);
% plot(xloc_u(:),ucloc_imag(:),'r--',xloc_v(:),vcloc_imag(:),'b--');
% legend({'u','v'},'Location','southeast')
% title('initial projection for the imaginary part')
% return
% % break point 1

% Time stepping - SSPRK(5,4)
CFL = 0.75; 
dt = CFL/(2*pi)*(min(h)^2);
%dt = 0.1*dt;
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
    energy(1,1) = (h(j)/2)*((Pv*v_real(:,j)).*Wv)'*(Pv*v_real(:,j));
    energy(1,1) = energy(1,1) + (h(j)/2)*((Pv*v_imag(:,j)).*Wv)'*(Pv*v_imag(:,j));
    
    energy(1,1) = energy(1,1) + (h(j)/2)*(((2/h(j))*S_u*u_real(:,j) - sin(xloc_u(:,j))).*Wu)'*...
        ((2/h(j))*S_u*u_real(:,j) - sin(xloc_u(:,j)));
    energy(1,1) = energy(1,1) + (h(j)/2)*(((2/h(j))*S_u*u_imag(:,j) + cos(xloc_u(:,j))).*Wu)'*...
        ((2/h(j))*S_u*u_imag(:,j) + cos(xloc_u(:,j)));
    
    energy(1,1) = energy(1,1) + (h(j)/2)*((Pu*u_real(:,j) + cos(xloc_u(:,j))).*Wu)'*...
        (Pu*u_real(:,j) + cos(xloc_u(:,j)));
    energy(1,1) = energy(1,1) + (h(j)/2)*((Pu*u_imag(:,j) + sin(xloc_u(:,j))).*Wu)'*...
        (Pu*u_imag(:,j) + sin(xloc_u(:,j)));
end

for it = 1:nsteps
    
    [rhsu_real, rhsv_real, rhsu_imag, rhsv_imag] = wave_schrodinger_update(q, N, h,...
        S_u, S_v, Sv_M, xloc_u, xloc_v, Wu, Wv, Pu, Pv, Pv_M, alpha, Vb_u, Vb_v, v_real, ...
        Uxb_u, Uxb_v, u_real, tau, beta, v_imag, u_imag);
    
    u_real1 = u_real + c11*dt*rhsu_real;
    v_real1 = v_real + c11*dt*rhsv_real;    
    u_imag1 = u_imag + c11*dt*rhsu_imag;
    v_imag1 = v_imag + c11*dt*rhsv_imag;
    
    [rhsu_real, rhsv_real, rhsu_imag, rhsv_imag] = wave_schrodinger_update(q, N, h,...
        S_u, S_v, Sv_M, xloc_u, xloc_v, Wu, Wv, Pu, Pv, Pv_M, alpha, Vb_u, Vb_v, v_real1, ...
        Uxb_u, Uxb_v, u_real1, tau, beta, v_imag1, u_imag1);
    
    u_real2 = c20*u_real + c21*u_real1 + c22*dt*rhsu_real;
    v_real2 = c20*v_real + c21*v_real1 + c22*dt*rhsv_real;   
    u_imag2 = c20*u_imag + c21*u_imag1 + c22*dt*rhsu_imag;
    v_imag2 = c20*v_imag + c21*v_imag1 + c22*dt*rhsv_imag;
    
    
    [rhsu_real, rhsv_real, rhsu_imag, rhsv_imag] = wave_schrodinger_update(q, N, h,...
        S_u, S_v, Sv_M, xloc_u, xloc_v, Wu, Wv, Pu, Pv, Pv_M, alpha, Vb_u, Vb_v, v_real2, ...
        Uxb_u, Uxb_v, u_real2, tau, beta, v_imag2, u_imag2);
    
    u_real3 = c30*u_real + c32*u_real2 + c33*dt*rhsu_real;
    v_real3 = c30*v_real + c32*v_real2 + c33*dt*rhsv_real;   
    u_imag3 = c30*u_imag + c32*u_imag2 + c33*dt*rhsu_imag;
    v_imag3 = c30*v_imag + c32*v_imag2 + c33*dt*rhsv_imag;
    
    
    [rhsu_real3, rhsv_real3, rhsu_imag3, rhsv_imag3] = wave_schrodinger_update(q, N, h,...
        S_u, S_v, Sv_M, xloc_u, xloc_v, Wu, Wv, Pu, Pv, Pv_M, alpha, Vb_u, Vb_v, v_real3, ...
        Uxb_u, Uxb_v, u_real3, tau, beta, v_imag3, u_imag3);
    
    u_real4 = c40*u_real + c43*u_real3 + c44*dt*rhsu_real3;
    v_real4 = c40*v_real + c43*v_real3 + c44*dt*rhsv_real3;   
    u_imag4 = c40*u_imag + c43*u_imag3 + c44*dt*rhsu_imag3;
    v_imag4 = c40*v_imag + c43*v_imag3 + c44*dt*rhsv_imag3;
    
    
    [rhsu_real, rhsv_real, rhsu_imag, rhsv_imag] = wave_schrodinger_update(q, N, h,...
        S_u, S_v, Sv_M, xloc_u, xloc_v, Wu, Wv, Pu, Pv, Pv_M, alpha, Vb_u, Vb_v, v_real4, ...
        Uxb_u, Uxb_v, u_real4, tau, beta, v_imag4, u_imag4);
    
    u_real = c52*u_real2 + c53*u_real3 + c53_1*dt*rhsu_real3 + c54*u_real4 + c55*dt*rhsu_real;
    v_real = c52*v_real2 + c53*v_real3 + c53_1*dt*rhsv_real3 + c54*v_real4 + c55*dt*rhsv_real;   
    u_imag = c52*u_imag2 + c53*u_imag3 + c53_1*dt*rhsu_imag3 + c54*u_imag4 + c55*dt*rhsu_imag;
    v_imag = c52*v_imag2 + c53*v_imag3 + c53_1*dt*rhsv_imag3 + c54*v_imag4 + c55*dt*rhsv_imag;
    
    for j=1:N
        energy(it+1,1) = (h(j)/2)*((Pv*v_real(:,j)).*Wv)'*(Pv*v_real(:,j));
        energy(it+1,1) = energy(it+1,1) + (h(j)/2)*((Pv*v_imag(:,j)).*Wv)'*(Pv*v_imag(:,j));
    
        energy(it+1,1) = energy(it+1,1) + (h(j)/2)*(((2/h(j))*S_u*u_real(:,j) - sin(xloc_u(:,j))).*Wu)'*...
            ((2/h(j))*S_u*u_real(:,j) - sin(xloc_u(:,j)));
        energy(it+1,1) = energy(it+1,1) + (h(j)/2)*(((2/h(j))*S_u*u_imag(:,j) + cos(xloc_u(:,j))).*Wu)'*...
            ((2/h(j))*S_u*u_imag(:,j) + cos(xloc_u(:,j)));
        
        energy(it+1,1) = energy(it+1,1) + (h(j)/2)*((Pu*u_real(:,j) + cos(xloc_u(:,j))).*Wu)'*...
            (Pu*u_real(:,j) + cos(xloc_u(:,j)));
        energy(it+1,1) = energy(it+1,1) + (h(j)/2)*((Pu*u_imag(:,j) + sin(xloc_u(:,j))).*Wu)'*...
            (Pu*u_imag(:,j) + sin(xloc_u(:,j)));
    end
    
    time(it+1,1) = time(it,1) + dt;
    
end


% Error check 

err_ureal = 0;
err_uimag = 0;
err_vreal = 0;
err_vimag = 0;
ucloc = zeros(q+1,N);
ucloc_real = ucloc;
ucloc_imag = ucloc;
grad_ucloc_real = ucloc;

vcloc_real = zeros(q,N);
vcloc_imag = vcloc_real;

utrue_real = ucloc;
utrue_imag = ucloc;
grad_utrue_real = ucloc;

vtrue_real = vcloc_real;
vtrue_imag = vcloc_real;

for j=1:N
    [uloc_u, grad_uloc_u, vloc_u] = soltrue(xloc_u(:,j),T);
    utrue_real(:,j) = (uloc_u(1,:)');
    utrue_imag(:,j) = (uloc_u(2,:)');
    grad_utrue_real(:,j) = (grad_uloc_u(1,:)');
    
    [uloc_v, grad_uloc_v, vloc_v] = soltrue(xloc_v(:,j),T);
    vtrue_real(:,j) = (vloc_v(1,:)');
    vtrue_imag(:,j) = (vloc_v(2,:)');
  
    ucloc_real(:,j) = (Pu*u_real(:,j)) + cos(xloc_u(:,j));
    ucloc_imag(:,j) = (Pu*u_imag(:,j)) + sin(xloc_u(:,j));
    grad_ucloc_real(:,j) = (2/h(j))*S_u*u_real(:,j) + sin(xloc_u(:,j));
    
    vcloc_real(:,j) = Pv*v_real(:,j);
    vcloc_imag(:,j) = Pv*v_imag(:,j);
    
    % L2
    err_ureal = err_ureal + (h(j)/2)*((ucloc_real(:,j) - utrue_real(:,j)).*Wu)'*(ucloc_real(:,j) - utrue_real(:,j));
    err_uimag = err_uimag + (h(j)/2)*((ucloc_imag(:,j) - utrue_imag(:,j)).*Wu)'*(ucloc_imag(:,j) - utrue_imag(:,j));
    err_vreal = err_vreal + (h(j)/2)*((vcloc_real(:,j) - vtrue_real(:,j)).*Wv)'*(vcloc_real(:,j) - vtrue_real(:,j));
    err_vimag = err_vimag + (h(j)/2)*((vcloc_imag(:,j) - vtrue_imag(:,j)).*Wv)'*(vcloc_imag(:,j) - vtrue_imag(:,j));

end

% plot(xloc_u(:),ucloc_real(:),'b--',xloc_u(:),utrue_real(:),'r--')
% plot(xloc_u(:),ucloc_real(:),'b--')

fprintf('real u L2 error = %4.3e \n',sqrt(err_ureal));
fprintf('imag u L2 error = %4.3e \n',sqrt(err_uimag));
fprintf('real v L2 error = %4.3e \n',sqrt(err_vreal));
fprintf('imag v L2 error = %4.3e \n',sqrt(err_vimag));

end

function [u, grad_u, v] = solnew(xloc,t)

u = [(cos(xloc+t)-cos(xloc))'; (sin(xloc+t)-sin(xloc))'];
grad_u = [(-sin(xloc+t)+sin(xloc))' ; (cos(xloc+t)-cos(xloc))'];
v = [-sin(xloc+t)'; cos(xloc+t)'];

end

function [u, grad_u, v] = soltrue(xloc,t)

u = [(cos(xloc+t))'; (sin(xloc+t))'];
grad_u = [(-sin(xloc+t))' ; (cos(xloc+t))'];
v = [-sin(xloc+t)'; cos(xloc+t)'];

end

function [ut_real, vt_real, ut_imag, vt_imag] = wave_schrodinger_update(q, N, h,...
        S_u, S_v, Sv_M, xloc_u, xloc_v, Wu, Wv, Pu, Pv, Pv_M, alpha, Vb_u, Vb_v, v_real, ...
        Uxb_u, Uxb_v, u_real, tau, beta, v_imag, u_imag)

ut_real = zeros(q+1,N);
vt_real = zeros(q,N);
ut_imag = zeros(q+1,N);
vt_imag = zeros(q,N);


for j = 1:N
            
            % assume u and v of the same order for now
            Mu = (2/h(j))*(S_u')*diag(Wu)*S_u + (h(j)/2)*(Pu')*diag(Wu)*Pu;
            Sv = -((2/h(j))*(S_u')*diag(Wu)*Sv_M + (h(j)/2)*(Pu')*diag(Wu)*Pv_M);
            Mv = (h(j)/2)*(Pv')*diag(Wv)*Pv;
            Su = (2/h(j))*(Sv_M')*diag(Wu)*S_u + (h(j)/2)*(Pv_M')*diag(Wu)*Pu;
            

            % Flux
            % periodic BC
            
            if (j==1)
                vstarl_real = alpha*(Vb_v(2,:)*(v_real(:,N)));
                vstarl_real = vstarl_real + (1-alpha)*(Vb_v(1,:)*(v_real(:,j)));
                jump = (2/h(N))*(Uxb_u(2,:)*(u_real(:,N))) - (2/h(j))*(Uxb_u(1,:)*(u_real(:,j)));
                vstarl_real = vstarl_real - tau*jump;
                
                vstarr_real = alpha*(Vb_v(2,:)*(v_real(:,j)));
                vstarr_real = vstarr_real + (1-alpha)*(Vb_v(1,:)*(v_real(:,j+1)));
                jump = (2/h(j))*(Uxb_u(2,:)*(u_real(:,j))) - (2/h(j+1))*(Uxb_u(1,:)*(u_real(:,j+1)));
                vstarr_real = vstarr_real - tau*jump;

                
                wstarl_real = (1-alpha)*((2/h(N))*Uxb_u(2,:)*(u_real(:,N)));
                wstarl_real = wstarl_real + alpha*((2/h(j))*Uxb_u(1,:)*(u_real(:,j)));
                jump = Vb_v(2,:)*(v_real(:,N)) - Vb_v(1,:)*(v_real(:,j));
                wstarl_real = wstarl_real - beta*jump;
                
                wstarr_real = (1-alpha)*((2/h(j))*Uxb_u(2,:)*(u_real(:,j)));
                wstarr_real = wstarr_real + alpha*((2/h(j+1))*Uxb_u(1,:)*(u_real(:,j+1)));
                jump = Vb_v(2,:)*(v_real(:,j)) - Vb_v(1,:)*(v_real(:,j+1));
                wstarr_real = wstarr_real - beta*jump;
                
                
                vstarl_imag = alpha*(Vb_v(2,:)*(v_imag(:,N)));
                vstarl_imag = vstarl_imag + (1-alpha)*(Vb_v(1,:)*(v_imag(:,j)));
                jump = (2/h(N))*Uxb_u(2,:)*(u_imag(:,N)) - (2/h(j))*(Uxb_u(1,:)*(u_imag(:,j)));
                vstarl_imag = vstarl_imag - tau*jump;
                
                vstarr_imag = alpha*(Vb_v(2,:)*(v_imag(:,j)));
                vstarr_imag = vstarr_imag + (1-alpha)*(Vb_v(1,:)*(v_imag(:,j+1)));
                jump = (2/h(j))*(Uxb_u(2,:)*(u_imag(:,j))) - (2/h(j+1))*(Uxb_u(1,:)*(u_imag(:,j+1)));
                vstarr_imag = vstarr_imag - tau*jump;
                
                
                wstarl_imag = (1-alpha)*((2/h(N))*Uxb_u(2,:)*(u_imag(:,N)));
                wstarl_imag = wstarl_imag + alpha*((2/h(j))*Uxb_u(1,:)*(u_imag(:,j)));
                jump = Vb_v(2,:)*(v_imag(:,N)) - Vb_v(1,:)*(v_imag(:,j));
                wstarl_imag = wstarl_imag - beta*jump;
                
                wstarr_imag = (1-alpha)*((2/h(j))*Uxb_u(2,:)*(u_imag(:,j)));
                wstarr_imag = wstarr_imag + alpha*((2/h(j+1))*Uxb_u(1,:)*(u_imag(:,j+1)));
                jump = Vb_v(2,:)*(v_imag(:,j)) - Vb_v(1,:)*(v_imag(:,j+1));
                wstarr_imag = wstarr_imag - beta*jump;
                
            elseif (j==N)
                
                vstarl_real = vstarr_real;

                vstarr_real = alpha*(Vb_v(2,:)*(v_real(:,j)));
                vstarr_real = vstarr_real + (1-alpha)*(Vb_v(1,:)*(v_real(:,1)));
                jump = ((2/h(j))*Uxb_u(2,:)*(u_real(:,j))) - ((2/h(1))*Uxb_u(1,:)*(u_real(:,1)));
                vstarr_real = vstarr_real - tau*jump;

                
                wstarl_real = wstarr_real;
                
                wstarr_real = (1-alpha)*((2/h(j))*Uxb_u(2,:)*(u_real(:,j)));
                wstarr_real = wstarr_real + alpha*((2/h(1))*Uxb_u(1,:)*(u_real(:,1)));
                jump = Vb_v(2,:)*(v_real(:,j)) - Vb_v(1,:)*(v_real(:,1));
                wstarr_real = wstarr_real - beta*jump;

                
                vstarl_imag = vstarr_imag;
                
                vstarr_imag = alpha*(Vb_v(2,:)*(v_imag(:,j)));
                vstarr_imag = vstarr_imag + (1-alpha)*(Vb_v(1,:)*(v_imag(:,1)));
                jump = ((2/h(j))*Uxb_u(2,:)*(u_imag(:,j))) - ((2/h(1))*Uxb_u(1,:)*(u_imag(:,1)));
                vstarr_imag = vstarr_imag - tau*jump;

                
                wstarl_imag = wstarr_imag;
                
                wstarr_imag = (1-alpha)*((2/h(j))*Uxb_u(2,:)*(u_imag(:,j)));
                wstarr_imag = wstarr_imag + alpha*((2/h(1))*Uxb_u(1,:)*(u_imag(:,1)));
                jump = Vb_v(2,:)*(v_imag(:,j)) - Vb_v(1,:)*(v_imag(:,1));
                wstarr_imag = wstarr_imag - beta*jump;

            else
                
                vstarl_real = vstarr_real;

                vstarr_real = alpha*(Vb_v(2,:)*(v_real(:,j)));
                vstarr_real = vstarr_real + (1-alpha)*(Vb_v(1,:)*(v_real(:,j+1)));
                jump = ((2/h(j))*Uxb_u(2,:)*(u_real(:,j))) - ((2/h(j+1))*Uxb_u(1,:)*(u_real(:,j+1)));
                vstarr_real = vstarr_real - tau*jump;

                                
                wstarl_real = wstarr_real;
                
                wstarr_real = (1-alpha)*((2/h(j))*Uxb_u(2,:)*(u_real(:,j)));
                wstarr_real = wstarr_real + alpha*((2/h(j+1))*Uxb_u(1,:)*(u_real(:,j+1)));
                jump = Vb_v(2,:)*(v_real(:,j)) - Vb_v(1,:)*(v_real(:,j+1));
                wstarr_real = wstarr_real - beta*jump;

                
                vstarl_imag = vstarr_imag;
                
                vstarr_imag = alpha*(Vb_v(2,:)*(v_imag(:,j)));
                vstarr_imag = vstarr_imag + (1-alpha)*(Vb_v(1,:)*(v_imag(:,j+1)));
                jump = ((2/h(j))*Uxb_u(2,:)*(u_imag(:,j))) - ((2/h(j+1))*Uxb_u(1,:)*(u_imag(:,j+1)));
                vstarr_imag = vstarr_imag - tau*jump;

                
                wstarl_imag = wstarr_imag;
                
                wstarr_imag = (1-alpha)*((2/h(j))*Uxb_u(2,:)*(u_imag(:,j)));
                wstarr_imag = wstarr_imag + alpha*((2/h(j+1))*Uxb_u(1,:)*(u_imag(:,j+1)));
                jump = Vb_v(2,:)*(v_imag(:,j)) - Vb_v(1,:)*(v_imag(:,j+1));
                wstarr_imag = wstarr_imag - beta*jump;
                
            end
            
            Fu_real = (-2/h(j))*(Uxb_u(2,:).*(Vb_v(2,:)*(v_real(:,j))) ...
                     - Uxb_u(1,:).*(Vb_v(1,:)*(v_real(:,j))));
            Fu_real = Fu_real + (2/h(j))*(Uxb_u(2,:).*vstarr_real - Uxb_u(1,:).*vstarl_real);
           
            ut_real(:,j) = -Sv*(v_real(:,j));
            ut_real(:,j) = ut_real(:,j) + (Fu_real');
            ut_real(:,j) = pinv(Mu)*(ut_real(:,j));
            
            
            Fv_real = (Vb_v(2,:).*wstarr_real - Vb_v(1,:).*wstarl_real);
            
            vt_real(:,j) = Mv*(v_imag(:,j))-Su*(u_real(:,j));
            vt_real(:,j) = vt_real(:,j) + (Fv_real');
            vt_real(:,j) = vt_real(:,j) - (h(j)/2)*(Pv')*(diag(Wv))*(2*cos(xloc_v(:,j)));
            vt_real(:,j) = Mv\(vt_real(:,j));
            
            
            Fu_imag = (-2/h(j))*(Uxb_u(2,:).*(Vb_v(2,:)*(v_imag(:,j))) ...
                     - Uxb_u(1,:).*(Vb_v(1,:)*(v_imag(:,j))));
            Fu_imag = Fu_imag + (2/h(j))*(Uxb_u(2,:).*vstarr_imag - Uxb_u(1,:).*vstarl_imag);

            ut_imag(:,j) = -Sv*(v_imag(:,j));
            ut_imag(:,j) = ut_imag(:,j) + (Fu_imag');
            ut_imag(:,j) = pinv(Mu)*(ut_imag(:,j));
            
            
            Fv_imag = (Vb_v(2,:).*wstarr_imag - Vb_v(1,:).*wstarl_imag);
            
            vt_imag(:,j) = -Mv*(v_real(:,j))-Su*(u_imag(:,j));
            vt_imag(:,j) = vt_imag(:,j) + (Fv_imag');
            vt_imag(:,j) = vt_imag(:,j) - (h(j)/2)*(Pv')*(diag(Wv))*(2*sin(xloc_v(:,j)));
            vt_imag(:,j) = Mv\(vt_imag(:,j));
            

end

end
