function [x,w] = GaussQCofs(q)

% Computes the Gauss-Legendre weights, w, and nodes, x, for a
% q-point formula on [-1,1] - requires q>1 

x=zeros(q,1);
w=zeros(q,1);

kk=(1:q-1)'.*(1:q-1)';

beta=sqrt(kk./(4*kk-1));
 
[V,D]=eig(diag(beta,1)+diag(beta,-1));

for i=1:q
  x(i)=D(i,i);
  w(i)=2*V(1,i)*V(1,i);
end

