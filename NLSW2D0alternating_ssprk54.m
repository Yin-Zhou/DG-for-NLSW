function [xloc,yloc,err_ureal,err_uimag,err_vreal,err_vimag] = NLSW2D0alternating_ssprk54(qx,qy,Nx,Ny,T)

% Solve u_tt - 1/2(u_xx + u_yy) + iu_t + u = 0 in 2D 0 < x < 2pi
% 0 < y < 2pi by DG
% but the initial condition is not 0, thus
% Solve u_tt - 1/2(u_xx + u_yy) + iu_t + u + 2e^{i(x+y)} = 0 in 2D 0 < x < 2pi
% 0 < y < 2pi by DG
% periodic boundary condition

% qx = degree for u in x-axis
% qy = degree for u in y-axis
% Nx = number of cells in x-axis
% Ny = number of cells in y-axis
% T is the simulation time
% alternating flux

% set up the grid

x = linspace(0,2*pi,Nx+1);
hx = x(2:Nx+1) - x(1:Nx);
y = linspace(0,2*pi,Ny+1);
hy = y(2:Ny+1) - y(1:Ny);

% construct the matrices

% get the nodes in reference domain [-1,1] and their weight
[rx,Wx] = GaussQCofs(qx+1);
[ry,Wy] = GaussQCofs(qy+1);

% get the qx+1*qx+1 matrix of legendre polynomials at nodes rx; the maximum degree is q 
Px = zeros(qx+1,qx+1);
for d = 1:qx+1
    Px(:,d) = JacobiP(rx,0,0,d-1);
end

Py = zeros(qy+1,qy+1);
for d = 1:qy+1
    Py(:,d) = JacobiP(ry,0,0,d-1);
end

% get the q+1 * q+1 matrix of the first derivative of legendre polynomials at nodes
% r; the maximum degree is q-1
Sx = zeros(qx+1,qx+1);
for d = 1:qx+1
    Sx(:,d) = GradJacobiP(rx,0,0,d-1);
end

Sy = zeros(qy+1,qy+1);
for d = 1:qy+1
    Sy(:,d) = GradJacobiP(ry,0,0,d-1);
end

% express the boundary [-1;1]
bp = [-1; 1];

Vb_x = zeros(2,qx+1); 
Uxb_x = Vb_x;
for d = 1:qx+1
    Vb_x(:,d) = JacobiP(bp,0,0,d-1);
    Uxb_x(:,d) = GradJacobiP(bp,0,0,d-1); 
end

Vb_y = zeros(2,qy+1); 
Uxb_y = Vb_y;
for d = 1:qy+1
    Vb_y(:,d) = JacobiP(bp,0,0,d-1);
    Uxb_y(:,d) = GradJacobiP(bp,0,0,d-1); 
end

% get the ((qx+1)*(qy+1)) * ((qx+1)*(qy+1)) matrix of legendre polynomials at nodes r; the maximum degree is q 
P = kron(Px,Py);
W = kron(diag(Wx),diag(Wy));

% assemble ut and vt (note you have all scaling in your formula, don't simplify it)
Mu = (1/2)*kron(Sx'*diag(Wx)*Sx, eye(qy+1)) + (1/2)*kron(eye(qx+1), Sy'*diag(Wy)*Sy) + (hx(1)/2)*(hy(1)/2)*eye((qx+1)*(qy+1));
Su = (1/2)*kron(eye(qx+1), Sy'*diag(Wy)*Sy) + (1/2)*kron(Sx'*diag(Wx)*Sx, eye(qy+1)) + (hx(1)/2)*(hy(1)/2)*eye((qx+1)*(qy+1));

% fluxes matrices

% Initialize u and v
u_real = zeros((qx+1)*(qy+1),(Nx*Ny));
u_imag = zeros((qx+1)*(qy+1),(Nx*Ny));
v_real = zeros((qx+1)*(qy+1),(Nx*Ny));
v_imag = zeros((qx+1)*(qy+1),(Nx*Ny));

% For initial data compute the L2 projections
xloc = zeros(qx+1,Nx);
yloc = zeros(qy+1,Ny);
for i=1:Nx
    xloc(:,i) = (x(i)+x(i+1)+hx(i)*rx)/2;
end

for j=1:Ny
    yloc(:,j) = (y(j)+y(j+1)+hy(j)*ry)/2;
end

for i=1:Nx
    for j=1:Ny
        
        [uloc, vloc] = solnew(xloc(:,i),yloc(:,j),0);

        for d = 1:((qx+1)*(qy+1))
            u_real(d,(i-1)*Ny+j) = (uloc(1,:)*(W))*P(:,d);
            v_real(d,(i-1)*Ny+j) = (vloc(1,:)*(W))*P(:,d);

            u_imag(d,(i-1)*Ny+j) = (uloc(2,:)*(W))*P(:,d); 
            v_imag(d,(i-1)*Ny+j) = (vloc(2,:)*(W))*P(:,d);
        end
    end
end

% % check the inital projection
% plot_ureal = P*u_real;
% plot_ureal_reshape = zeros((qx+1)*(qy+1),Nx*Ny);
% for j=1:Ny
%     for i=1:Nx
%         plot_ureal_reshape(:,(j-1)*Ny+i) = plot_ureal(:,(i-1)*Ny+j);
%     end
% end
% plot_z = zeros((qx+1)*Nx,(qy+1)*Ny);
% for j=1:Ny
%     for i=1:Nx
%         for k=1:(qy+1)
%             new = plot_ureal_reshape(k,(j-1)*Nx+i);
%             for l=2:(qx+1)
%                 new(l) = plot_ureal_reshape(k+(l-1)*(qy+1),(j-1)*Nx+i);
%             end
%             plot_z(((i-1)*(qx+1)+1):((i-1)*(qx+1)+qx+1), (j-1)*(qy+1)+k) = new;
%         end
%     end
% end
% plot_x = reshape(xloc,[],1);
% plot_y = reshape(yloc,[],1);
% % disp(size(plot_x));
% % disp(size(plot_y));
% % disp(size(plot_z));
% [X, Y] = meshgrid(plot_x, plot_y);
% mesh(X, Y, plot_z);
% colorbar
% title('proj')
% stop

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Time stepping - SSPRK(5,4)
CFL = 0.75;
dt = CFL/(2*pi)*(min(hx));
dt = 0.1*dt;
nsteps = ceil(T/dt);
disp(nsteps)
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

for it = 1:nsteps
    
    [rhsu_real, rhsv_real, rhsu_imag, rhsv_imag] = wave_schrodinger_update(qx, qy, hx, hy, Nx, Ny, Mu, Su, Vb_x, Vb_y, v_real, Uxb_x, Uxb_y, u_real, ...
        v_imag, u_imag, P, W, xloc, yloc);
    u_real1 = u_real + c11*dt*rhsu_real;
    v_real1 = v_real + c11*dt*rhsv_real;    
    u_imag1 = u_imag + c11*dt*rhsu_imag;
    v_imag1 = v_imag + c11*dt*rhsv_imag;
    
    [rhsu_real, rhsv_real, rhsu_imag, rhsv_imag] = wave_schrodinger_update(qx, qy, hx, hy, Nx, Ny, Mu, Su, Vb_x, Vb_y, v_real1, Uxb_x, Uxb_y, u_real1, ...
        v_imag1, u_imag1, P, W, xloc, yloc);
    u_real2 = c20*u_real + c21*u_real1 + c22*dt*rhsu_real;
    v_real2 = c20*v_real + c21*v_real1 + c22*dt*rhsv_real;   
    u_imag2 = c20*u_imag + c21*u_imag1 + c22*dt*rhsu_imag;
    v_imag2 = c20*v_imag + c21*v_imag1 + c22*dt*rhsv_imag;
    
    
    [rhsu_real, rhsv_real, rhsu_imag, rhsv_imag] = wave_schrodinger_update(qx, qy, hx, hy, Nx, Ny, Mu, Su, Vb_x, Vb_y, v_real2, Uxb_x, Uxb_y, u_real2, ...
        v_imag2, u_imag2, P, W, xloc, yloc);
    u_real3 = c30*u_real + c32*u_real2 + c33*dt*rhsu_real;
    v_real3 = c30*v_real + c32*v_real2 + c33*dt*rhsv_real;   
    u_imag3 = c30*u_imag + c32*u_imag2 + c33*dt*rhsu_imag;
    v_imag3 = c30*v_imag + c32*v_imag2 + c33*dt*rhsv_imag;
    
    
    [rhsu_real3, rhsv_real3, rhsu_imag3, rhsv_imag3] = wave_schrodinger_update(qx, qy, hx, hy, Nx, Ny, Mu, Su, Vb_x, Vb_y, v_real3, Uxb_x, Uxb_y, u_real3, ...
        v_imag3, u_imag3, P, W, xloc, yloc);
    u_real4 = c40*u_real + c43*u_real3 + c44*dt*rhsu_real3;
    v_real4 = c40*v_real + c43*v_real3 + c44*dt*rhsv_real3;   
    u_imag4 = c40*u_imag + c43*u_imag3 + c44*dt*rhsu_imag3;
    v_imag4 = c40*v_imag + c43*v_imag3 + c44*dt*rhsv_imag3;
    
    
    [rhsu_real, rhsv_real, rhsu_imag, rhsv_imag] = wave_schrodinger_update(qx, qy, hx, hy, Nx, Ny, Mu, Su, Vb_x, Vb_y, v_real4, Uxb_x, Uxb_y, u_real4, ...
        v_imag4, u_imag4, P, W, xloc, yloc);
    u_real = c52*u_real2 + c53*u_real3 + c53_1*dt*rhsu_real3 + c54*u_real4 + c55*dt*rhsu_real;
    v_real = c52*v_real2 + c53*v_real3 + c53_1*dt*rhsv_real3 + c54*v_real4 + c55*dt*rhsv_real;   
    u_imag = c52*u_imag2 + c53*u_imag3 + c53_1*dt*rhsu_imag3 + c54*u_imag4 + c55*dt*rhsu_imag;
    v_imag = c52*v_imag2 + c53*v_imag3 + c53_1*dt*rhsv_imag3 + c54*v_imag4 + c55*dt*rhsv_imag;
    
end

% Error check
err_ureal = 0;
err_uimag = 0;
err_vreal = 0;
err_vimag = 0;
ucloc = zeros(((qx+1)*(qy+1)),(Nx*Ny));
ucloc_real = ucloc;
ucloc_imag = ucloc;
utrue_real = ucloc;
utrue_imag = ucloc;

vcloc_real = ucloc;
vcloc_imag = ucloc;
vtrue_real = ucloc;
vtrue_imag = ucloc;

% uexact = zeros(Nx*(qx+1),Ny*(qy+1));

for i=1:Nx
    for j=1:Ny
        [uloc, vloc] = soltrue(xloc(:,i),yloc(:,j),T);
        
        utrue_real(:,(i-1)*Ny+j) = (uloc(1,:)');
        utrue_imag(:,(i-1)*Ny+j) = (uloc(2,:)');
        
        vtrue_real(:,(i-1)*Ny+j) = (vloc(1,:)');
        vtrue_imag(:,(i-1)*Ny+j) = (vloc(2,:)');
        
        cosine = zeros((qx+1)*(qy+1),1);
        sine = zeros((qx+1)*(qy+1),1);
        for k=1:(qx+1)
            cosine(((k-1)*(qy+1)+1):((k-1)*(qy+1)+(qy+1)),1) = cos(xloc(k,i)+yloc(:,j));
            sine(((k-1)*(qy+1)+1):((k-1)*(qy+1)+(qy+1)),1) = sin(xloc(k,i)+yloc(:,j));
        end
        
        ucloc_real(:,(i-1)*Ny+j) = (P*u_real(:,(i-1)*Ny+j)) + cosine;
        ucloc_imag(:,(i-1)*Ny+j) = (P*u_imag(:,(i-1)*Ny+j)) + sine;

        vcloc_real(:,(i-1)*Ny+j) = (P*v_real(:,(i-1)*Ny+j));
        vcloc_imag(:,(i-1)*Ny+j) = (P*v_imag(:,(i-1)*Ny+j));
        
        err_ureal = err_ureal + (hx(i)/2)*(hy(j)/2)*((ucloc_real(:,(i-1)*Ny+j) - ...
            utrue_real(:,(i-1)*Ny+j)))'*(W)*(ucloc_real(:,(i-1)*Ny+j) - utrue_real(:,(i-1)*Ny+j));

        err_uimag = err_uimag + (hx(i)/2)*(hy(j)/2)*((ucloc_imag(:,(i-1)*Ny+j) - ...
            utrue_imag(:,(i-1)*Ny+j)))'*(W)*(ucloc_imag(:,(i-1)*Ny+j) - utrue_imag(:,(i-1)*Ny+j));

        err_vreal = err_vreal + (hx(i)/2)*(hy(j)/2)*((vcloc_real(:,(i-1)*Ny+j) - ...
            vtrue_real(:,(i-1)*Ny+j)))'*(W)*(vcloc_real(:,(i-1)*Ny+j) - vtrue_real(:,(i-1)*Ny+j));

        err_vimag = err_vimag + (hx(i)/2)*(hy(j)/2)*((vcloc_imag(:,(i-1)*Ny+j) - ...
            vtrue_imag(:,(i-1)*Ny+j)))'*(W)*(vcloc_imag(:,(i-1)*Ny+j) - vtrue_imag(:,(i-1)*Ny+j));
    end
end

fprintf('u real L2 error = %4.3e \n',sqrt(err_ureal));
fprintf('u imag L2 error = %4.3e \n',sqrt(err_uimag));
fprintf('v real L2 error = %4.3e \n',sqrt(err_vreal));
fprintf('v imag L2 error = %4.3e \n',sqrt(err_vimag));

%plot_ureal = P*u_real;
%plot_ureal_reshape = zeros((qx+1)*(qy+1),Nx*Ny);
%for j=1:Ny
%    for i=1:Nx
%        plot_ureal_reshape(:,(j-1)*Ny+i) = plot_ureal(:,(i-1)*Ny+j);
%    end
%end
%plot_z = zeros((qx+1)*Nx,(qy+1)*Ny);
%for j=1:Ny
%    for i=1:Nx
%        for k=1:(qy+1)
%            new = plot_ureal_reshape(k,(j-1)*Nx+i);
%            for l=2:(qx+1)
%                new(l) = plot_ureal_reshape(k+(l-1)*(qy+1),(j-1)*Nx+i);
%            end
%            plot_z(((i-1)*(qx+1)+1):((i-1)*(qx+1)+qx+1), (j-1)*(qy+1)+k) = new;
%        end
%    end
%end
%plot_x = reshape(xloc,[],1);
%plot_y = reshape(yloc,[],1);
%[X, Y] = meshgrid(plot_x, plot_y);
%mesh(X, Y, plot_z);
%colorbar
%title('result')

end


function [u, v] = soltrue(xloc,yloc,t)
dim_x = size(xloc);
len_x = dim_x(1); % qx+1
dim_y = size(yloc);
len_y = dim_y(1); % qy+1

u = zeros(2,len_x*len_y);
v = zeros(2,len_x*len_y);

for i=1:len_x
    u(1,((i-1)*len_y+1):(i*len_y)) = (cos(xloc(i)+yloc+t))';
    u(2,((i-1)*len_y+1):(i*len_y)) = sin(xloc(i)+yloc+t)';
    
    v(1,((i-1)*len_y+1):(i*len_y)) = -sin(xloc(i)+yloc+t)';
    v(2,((i-1)*len_y+1):(i*len_y)) = cos(xloc(i)+yloc+t)';
end

end

function [u, v] = solnew(xloc,yloc,t)
dim_x = size(xloc);
len_x = dim_x(1); % qx+1
dim_y = size(yloc);
len_y = dim_y(1); % qy+1

u = zeros(2,len_x*len_y);
v = zeros(2,len_x*len_y);

for i=1:len_x
    u(1,((i-1)*len_y+1):(i*len_y)) = (cos(xloc(i)+yloc+t) - cos(xloc(i)+yloc))';
    u(2,((i-1)*len_y+1):(i*len_y)) = (sin(xloc(i)+yloc+t) - sin(xloc(i)+yloc))';
    
    v(1,((i-1)*len_y+1):(i*len_y)) = -sin(xloc(i)+yloc+t)';
    v(2,((i-1)*len_y+1):(i*len_y)) = cos(xloc(i)+yloc+t)';
end

end

function [ut_real, vt_real, ut_imag, vt_imag] = wave_schrodinger_update(qx, qy, hx, hy, Nx, Ny, Mu, Su, Vb_x, Vb_y, v_real, Uxb_x, Uxb_y, u_real, ...
    v_imag, u_imag, P, W, xloc, yloc)

ut_real = zeros((qx+1)*(qy+1),(Nx*Ny));
ut_imag = zeros((qx+1)*(qy+1),(Nx*Ny));
vt_real = zeros((qx+1)*(qy+1),(Nx*Ny));
vt_imag = zeros((qx+1)*(qy+1),(Nx*Ny));

for i=1:Nx
    for j=1:Ny
        
        % local value coressponds to (i,j)
        vreal_local = v_real(:, (i-1)*Ny + j);
        ureal_local = u_real(:, (i-1)*Ny + j);
        vimag_local = v_imag(:, (i-1)*Ny + j);
        uimag_local = u_imag(:, (i-1)*Ny + j);
        
        
        ut_real(:,(i-1)*Ny+j) = vreal_local;
        ut_imag(:,(i-1)*Ny+j) = vimag_local;
        
        vt_real(:,(i-1)*Ny+j) = vimag_local - (2/hx(i))*(2/hy(j))*(Su*ureal_local);
        vt_imag(:,(i-1)*Ny+j) =-vreal_local - (2/hx(i))*(2/hy(j))*(Su*uimag_local);
        
        cosine = zeros((qx+1)*(qy+1),1);
        sine = zeros((qx+1)*(qy+1),1);
        for k=1:(qx+1)
            cosine(((k-1)*(qy+1)+1):((k-1)*(qy+1)+qy+1),1) = cos(xloc(k,i)+yloc(:,j));
            sine(((k-1)*(qy+1)+1):((k-1)*(qy+1)+qy+1),1) = sin(xloc(k,i)+yloc(:,j));
        end
        vt_real(:,(i-1)*Ny+j) = vt_real(:,(i-1)*Ny+j) - (P')*W*2*cosine;
        vt_imag(:,(i-1)*Ny+j) = vt_imag(:,(i-1)*Ny+j) - (P')*W*2*sine;
        
        % alternating flux: star = local at up and right; out at bottom and left
        % periodic BC
        % (i-1)*Ny + j
        if (j==1) % out corresponds to j == Ny (i-1)*Ny + Ny south side
           vreal_out = v_real(:, (i-1)*Ny + Ny);
           ureal_out = u_real(:, (i-1)*Ny + Ny);
           vimag_out = v_imag(:, (i-1)*Ny + Ny);
           uimag_out = u_imag(:, (i-1)*Ny + Ny);
        else % out corresponds to j == j-1 (i-1)*Ny + j - 1
           vreal_out = v_real(:, (i-1)*Ny + j-1);
           ureal_out = u_real(:, (i-1)*Ny + j-1);
           vimag_out = v_imag(:, (i-1)*Ny + j-1);
           uimag_out = u_imag(:, (i-1)*Ny + j-1);
        end
            
        Fu_real = - kron(eye(qx+1), (Uxb_y(1,:)'*Vb_y(1,:)))*(-vreal_local)...
                  - kron(eye(qx+1), (Uxb_y(1,:)'*Vb_y(2,:)))*( vreal_out);
              
        Fu_imag = - kron(eye(qx+1), (Uxb_y(1,:)'*Vb_y(1,:)))*(-vimag_local)...
                  - kron(eye(qx+1), (Uxb_y(1,:)'*Vb_y(2,:)))*( vimag_out);
       
        Fv_real = - kron(eye(qx+1), (Vb_y(1,:)'*Uxb_y(1,:)))*(ureal_local);
        
        Fv_imag = - kron(eye(qx+1), (Vb_y(1,:)'*Uxb_y(1,:)))*(uimag_local);
         
        if (j==Ny) % out corresponds to j == 1 north side
           vreal_out = v_real(:, (i-1)*Ny + 1);
           ureal_out = u_real(:, (i-1)*Ny + 1);
           vimag_out = v_imag(:, (i-1)*Ny + 1);
           uimag_out = u_imag(:, (i-1)*Ny + 1);
        else % out corresponds to j + 1
           vreal_out = v_real(:, (i-1)*Ny + j+1);
           ureal_out = u_real(:, (i-1)*Ny + j+1);
           vimag_out = v_imag(:, (i-1)*Ny + j+1);
           uimag_out = u_imag(:, (i-1)*Ny + j+1);
        end
        
        Fv_real = Fv_real + kron(eye(qx+1), (Vb_y(2,:)'*Uxb_y(1,:)))*(ureal_out);

        Fv_imag = Fv_imag + kron(eye(qx+1), (Vb_y(2,:)'*Uxb_y(1,:)))*(uimag_out);

        if (i==1) % out corresponds to i = Nx west side
           vreal_out = v_real(:, (Nx-1)*Ny + j);
           ureal_out = u_real(:, (Nx-1)*Ny + j);
           vimag_out = v_imag(:, (Nx-1)*Ny + j);
           uimag_out = u_imag(:, (Nx-1)*Ny + j);
        else % out corresponds to i - 1
           vreal_out = v_real(:, (i-2)*Ny + j);
           ureal_out = u_real(:, (i-2)*Ny + j);
           vimag_out = v_imag(:, (i-2)*Ny + j);
           uimag_out = u_imag(:, (i-2)*Ny + j);
        end
        
        Fu_real = Fu_real - kron((Uxb_x(1,:)'*Vb_x(1,:)), eye(qy+1))*(-vreal_local);
        Fu_real = Fu_real - kron((Uxb_x(1,:)'*Vb_x(2,:)), eye(qy+1))*( vreal_out);
      
        Fu_imag = Fu_imag - kron((Uxb_x(1,:)'*Vb_x(1,:)), eye(qy+1))*(-vimag_local);
        Fu_imag = Fu_imag - kron((Uxb_x(1,:)'*Vb_x(2,:)), eye(qy+1))*( vimag_out);
        
        Fv_real = Fv_real - kron((Vb_x(1,:)'*Uxb_x(1,:)), eye(qy+1))*(ureal_local);

        Fv_imag = Fv_imag - kron((Vb_x(1,:)'*Uxb_x(1,:)), eye(qy+1))*(uimag_local);
        
        if (i==Nx) % out corresponds to i = 1 east side
           vreal_out = v_real(:, j);
           ureal_out = u_real(:, j);
           vimag_out = v_imag(:, j);
           uimag_out = u_imag(:, j);
        else % out corresponds to i + 1
           vreal_out = v_real(:, i*Ny + j);
           ureal_out = u_real(:, i*Ny + j);
           vimag_out = v_imag(:, i*Ny + j);
           uimag_out = u_imag(:, i*Ny + j);
        end
        
        Fv_real = Fv_real + kron((Vb_x(2,:)'*Uxb_x(1,:)), eye(qy+1))*(ureal_out);
        
        Fv_imag = Fv_imag + kron((Vb_x(2,:)'*Uxb_x(1,:)), eye(qy+1))*(uimag_out);

        ut_real(:,(i-1)*Ny+j) = ut_real(:,(i-1)*Ny+j) + Mu\((1/2)*Fu_real);
        ut_imag(:,(i-1)*Ny+j) = ut_imag(:,(i-1)*Ny+j) + Mu\((1/2)*Fu_imag);
        
        vt_real(:,(i-1)*Ny+j) = vt_real(:,(i-1)*Ny+j) + (2/hx(i))*(2/hy(j))*((1/2)*Fv_real);
        vt_imag(:,(i-1)*Ny+j) = vt_imag(:,(i-1)*Ny+j) + (2/hx(i))*(2/hy(j))*((1/2)*Fv_imag);        
    end
end

end
