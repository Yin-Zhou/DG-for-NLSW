function [xloc,utrue,ucloc,err_ureal,energy,time] = NLSWflux(q,N,T,alpha,tau,beta)

% Solve u_tt - u_xx + iu_t + u = 0 in 1D 0 < x < 2pi by DG

% q = degree for u
% q = degree for v for now, can also choose q-1
% N = number of cells
% T is the simulation time
% alpha,tau,beta for the numerical flux

% set up the grid

x = linspace(0,2*pi,N+1);
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
ut_real = zeros(q+1,N);
ut_imag = zeros(q+1,N);
v_real = zeros(q+1,N);
v_imag = zeros(q+1,N);
vt_real = zeros(q+1,N);
vt_imag = zeros(q+1,N);

% For initial data compute the L2 projections 
xloc = zeros(q+1,N);
for j=1:N
    xloc(:,j) = (x(j)+x(j+1)+h(j)*r)/2;  
    [uloc, vloc] = solnew(xloc(:,j),0);

    for d = 1:q+1
        u_real(d,j) = (uloc(1,:).*(W'))*P(:,d);
        v_real(d,j) = (vloc(1,:).*(W'))*P(:,d);
    
        u_imag(d,j) = (uloc(2,:).*(W'))*P(:,d); 
        v_imag(d,j) = (vloc(2,:).*(W'))*P(:,d);
    end
end

% % plot the inital projection
% figure

% % disp(size(xloc(:)));
% % disp(xloc(:)/2/pi);

% % top plot
% subplot(2,1,1);
% plot(xloc,P*u_real,xloc,P*v_real);
% % legend({'u_r','v_r'},'Location','southeast')
% title('initial projection for the real part')
% 
% % bottom plot
% subplot(2,1,2);
% plot(xloc,P*u_imag,xloc,P*v_imag);
% % legend({'u_i','v_i'},'Location','southeast')
% title('initial projection for the imaginary part')

% break point 1
% return

% Time stepping - order q Taylor series
dt = (min(h)^2)/20;
nsteps = round(T/dt);
dt = T/nsteps;
% disp(dt);
% disp(nsteps);

up_real = u_real;
vp_real = v_real;
    
up_imag = u_imag;
vp_imag = v_imag;

% check energy conservation of the time integrator
energy = zeros(nsteps+1,1);
time = zeros(nsteps+1,1);
for j=1:N
    energy(1,1) = (h(j)/2)*((P*v_real(:,j)).*W)'*(P*v_real(:,j));
    energy(1,1) = energy(1,1) + (h(j)/2)*((P*v_imag(:,j)).*W)'*(P*v_imag(:,j));
    
    energy(1,1) = energy(1,1) + (h(j)/2)*(((2/h(j))*S*u_real(:,j)).*W)'*((2/h(j))*S*u_real(:,j));
    energy(1,1) = energy(1,1) + (h(j)/2)*(((2/h(j))*S*u_imag(:,j)).*W)'*((2/h(j))*S*u_imag(:,j));
end

for it = 1:nsteps
    
    dts = dt;
    
    for ist=1:q
        for j = 1:N
            
            % assume u and v of the same order for now
            
            Mu = (2/h(j))*(S')*diag(W)*S + (h(j)/2)*(P')*diag(W)*P;
            Sv = -Mu;
            Mv = (h(j)/2)*(P')*diag(W)*P;
            Su = Mu;
            

            % Flux
            % periodic BC
            
            if (j==1)
                vstarl_real = alpha*(Vb(2,:)*(v_real(:,N)));
                vstarl_real = vstarl_real + (1-alpha)*(Vb(1,:)*(v_real(:,j)));
                jump = (Uxb(2,:)*(u_real(:,N))) - (Uxb(1,:)*(u_real(:,j)));
                vstarl_real = vstarl_real - tau*jump;
                
                vstarr_real = alpha*(Vb(2,:)*(v_real(:,j)));
                vstarr_real = vstarr_real + (1-alpha)*(Vb(1,:)*(v_real(:,j+1)));
                jump = (Uxb(2,:)*(u_real(:,j))) - (Uxb(1,:)*(u_real(:,j+1)));
                vstarr_real = vstarr_real - tau*jump;

                
                wstarl_real = (1-alpha)*(Uxb(2,:)*(u_real(:,N)));
                wstarl_real = wstarl_real + alpha*(Uxb(1,:)*(u_real(:,j)));
                jump = Vb(2,:)*(v_real(:,N)) - Vb(1,:)*(v_real(:,j));
                wstarl_real = wstarl_real - beta*jump;
                
                wstarr_real = (1-alpha)*(Uxb(2,:)*(u_real(:,j)));
                wstarr_real = wstarr_real + alpha*(Uxb(1,:)*(u_real(:,j+1)));
                jump = Vb(2,:)*(v_real(:,j)) - Vb(1,:)*(v_real(:,j+1));
                wstarr_real = wstarr_real - beta*jump;
                
                
                vstarl_imag = alpha*(Vb(2,:)*(v_imag(:,N)));
                vstarl_imag = vstarl_imag + (1-alpha)*(Vb(1,:)*(v_imag(:,j)));
                jump = (Uxb(2,:)*(u_imag(:,N))) - (Uxb(1,:)*(u_imag(:,j)));
                vstarl_imag = vstarl_imag - tau*jump;
                
                vstarr_imag = alpha*(Vb(2,:)*(v_imag(:,j)));
                vstarr_imag = vstarr_imag + (1-alpha)*(Vb(1,:)*(v_imag(:,j+1)));
                jump = (Uxb(2,:)*(u_imag(:,j))) - (Uxb(1,:)*(u_imag(:,j+1)));
                vstarr_imag = vstarr_imag - tau*jump;
                
                
                wstarl_imag = (1-alpha)*(Uxb(2,:)*(u_imag(:,N)));
                wstarl_imag = wstarl_imag + alpha*(Uxb(1,:)*(u_imag(:,j)));
                jump = Vb(2,:)*(v_imag(:,N)) - Vb(1,:)*(v_imag(:,j));
                wstarl_imag = wstarl_imag - beta*jump;
                
                wstarr_imag = (1-alpha)*(Uxb(2,:)*(u_imag(:,j)));
                wstarr_imag = wstarr_imag + alpha*(Uxb(1,:)*(u_imag(:,j+1)));
                jump = Vb(2,:)*(v_imag(:,j)) - Vb(1,:)*(v_imag(:,j+1));
                wstarr_imag = wstarr_imag - beta*jump;
                
            elseif (j==N)
                
                vstarl_real = vstarr_real;

                vstarr_real = alpha*(Vb(2,:)*(v_real(:,j)));
                vstarr_real = vstarr_real + (1-alpha)*(Vb(1,:)*(v_real(:,1)));
                jump = (Uxb(2,:)*(u_real(:,j))) - (Uxb(1,:)*(u_real(:,1)));
                vstarr_real = vstarr_real - tau*jump;

                
                wstarl_real = wstarr_real;
                
                wstarr_real = (1-alpha)*(Uxb(2,:)*(u_real(:,j)));
                wstarr_real = wstarr_real + alpha*(Uxb(1,:)*(u_real(:,1)));
                jump = Vb(2,:)*(v_real(:,j)) - Vb(1,:)*(v_real(:,1));
                wstarr_real = wstarr_real - beta*jump;

                
                vstarl_imag = vstarr_imag;
                
                vstarr_imag = alpha*(Vb(2,:)*(v_imag(:,j)));
                vstarr_imag = vstarr_imag + (1-alpha)*(Vb(1,:)*(v_imag(:,1)));
                jump = (Uxb(2,:)*(u_imag(:,j))) - (Uxb(1,:)*(u_imag(:,1)));
                vstarr_imag = vstarr_imag - tau*jump;

                
                wstarl_imag = wstarr_imag;
                
                wstarr_imag = (1-alpha)*(Uxb(2,:)*(u_imag(:,j)));
                wstarr_imag = wstarr_imag + alpha*(Uxb(1,:)*(u_imag(:,1)));
                jump = Vb(2,:)*(v_imag(:,j)) - Vb(1,:)*(v_imag(:,1));
                wstarr_imag = wstarr_imag - beta*jump;

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
            ut_real(:,j) = Mu\(ut_real(:,j));
            
            
            Fv_real = (2/h(j))*(Vb(2,:).*wstarr_real - Vb(1,:).*wstarl_real);
            
            vt_real(:,j) = Mv*(v_imag(:,j))-Su*(u_real(:,j));
            vt_real(:,j) = vt_real(:,j) + (Fv_real');
            vt_real(:,j) = Mv\(vt_real(:,j));
            
            
            Fu_imag = (-2/h(j))*(Uxb(2,:).*(Vb(2,:)*(v_imag(:,j))) ...
                     - Uxb(1,:).*(Vb(1,:)*(v_imag(:,j))));
            Fu_imag = Fu_imag + (2/h(j))*(Uxb(2,:).*vstarr_imag - Uxb(1,:).*vstarl_imag);

            ut_imag(:,j) = -Sv*(v_imag(:,j));
            ut_imag(:,j) = ut_imag(:,j) + (Fu_imag');
            ut_imag(:,j) = Mu\(ut_imag(:,j));
            
            
            Fv_imag = (2/h(j))*(Vb(2,:).*wstarr_imag - Vb(1,:).*wstarl_imag);
            
            vt_imag(:,j) = -Mv*(v_real(:,j))-Su*(u_imag(:,j));
            vt_imag(:,j) = vt_imag(:,j) + (Fv_imag');
            vt_imag(:,j) = Mv\(vt_imag(:,j));
            

        end
        
        up_real = up_real + dts*(ut_real);
        vp_real = vp_real + dts*(vt_real);
        u_real = ut_real;
        v_real = vt_real;

        up_imag = up_imag + dts*(ut_imag);
        vp_imag = vp_imag + dts*(vt_imag);
        u_imag = ut_imag;
        v_imag = vt_imag;
     
        
        dts = dts*(dt/(ist+1));
    end
    
    u_real = up_real;
    v_real = vp_real;
    
    u_imag = up_imag;
    v_imag = vp_imag;
    
    for j=1:N
        energy(it+1,1) = (h(j)/2)*((P*v_real(:,j)).*W)'*(P*v_real(:,j));
        energy(it+1,1) = energy(it+1,1) + (h(j)/2)*((P*v_imag(:,j)).*W)'*(P*v_imag(:,j));
    
        energy(it+1,1) = energy(it+1,1) + (h(j)/2)*(((2/h(j))*S*u_real(:,j)).*W)'*((2/h(j))*S*u_real(:,j));
        energy(it+1,1) = energy(it+1,1) + (h(j)/2)*(((2/h(j))*S*u_imag(:,j)).*W)'*((2/h(j))*S*u_imag(:,j));
    end
    
    time(it+1,1) = time(it,1) + dt;
    
end


% Error check 

err_ureal = 0;
ucloc = zeros(q+1,N);
vcloc = ucloc;
utrue = ucloc;
vtrue = ucloc;
grad_utrue = ucloc;
grad_ucloc = ucloc;

for j=1:N
    [uloc, grad_uloc, vloc] = solnew(xloc(:,j),T);
    utrue(:,j) = (uloc(1,:)');
    grad_utrue(:,j) = (grad_uloc(1,:)');
  
% %   test for real part of v
%   vtrue(:,j) = (vloc(1,:)');
%   vcloc(:,j) = P*v_real(:,j);
%   err_ureal = err_ureal + (h(j)/2)*((vcloc(:,j) - vtrue(:,j)).*W)'*(vcloc(:,j) - vtrue(:,j));
  
% %   test for real part of u, H1 norm
    ucloc(:,j) = P*u_real(:,j);
    grad_ucloc(:,j) = (2/h(j))*S*u_real(:,j);
    err_ureal = err_ureal + (h(j)/2)*((ucloc(:,j) - utrue(:,j)).*W)'*(ucloc(:,j) - utrue(:,j));
    err_ureal = err_ureal + (h(j)/2)...
        *((grad_ucloc(:,j) - grad_utrue(:,j)).*W)'*(grad_ucloc(:,j) - grad_utrue(:,j));

end

fprintf('L2 error = %4.3e \n',sqrt(err_ureal));

end

function [u, grad_u, v] = solnew(xloc,t)

u = [cos(xloc+t)'; sin(xloc+t)'];
grad_u = [-sin(xloc+t)' ; cos(xloc+t)'];
v = [-sin(xloc+t)'; cos(xloc+t)'];

end