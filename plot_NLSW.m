% plot
errors = zeros(7,13);
meshwidth = zeros(7,13);
energys = zeros(7,50001);
for q=1:7
    for i=1:13
        N = 40+(i-1)*5;
        [xloc,utrue,ucloc,err,energy,time] = NLSWflux(q,N,pi^2,0.5,0,0);
        errors(q,i) = sqrt(err);
        meshwidth(q,i) = 2*pi/N;
        if (i==13)
           energys(q,:) = (energy'); 
        end
    end
        
    rate = polyfit(log(meshwidth(q,:)),log(errors(q,:)),1);
    fprintf('%d %d convergence rate = %4.3e \n',q,N,rate(1));
end
figure
loglog(meshwidth(1,:),errors(1,:),'--o');
hold on
loglog(meshwidth(2,:),errors(2,:),'--o');
hold on
loglog(meshwidth(3,:),errors(3,:),'--o');
hold on
loglog(meshwidth(4,:),errors(4,:),'--o');
hold on
loglog(meshwidth(5,:),errors(5,:),'--o');
hold on
loglog(meshwidth(6,:),errors(6,:),'--o');
hold on
loglog(meshwidth(7,:),errors(7,:),'--o');
xlabel('h');
ylabel('||e_u||_{H_1}')
legend({'q=1','q=2','q=3','q=4','q=5','q=6','q=7'});
title('H1 norm of u for the central flux')


figure
plot(time,energy(1,:),'--o');
hold on
plot(time,energy(2,:),'--o');
hold on
plot(time,energy(3,:),'--o');
hold on
plot(time,energy(4,:),'--o');
hold on
plot(time,energy(5,:),'--o');
hold on
plot(time,energy(6,:),'--o');
hold on
plot(time,energy(7,:),'--o');
xlabel('t');
ylabel('energy')
legend({'q=1','q=2','q=3','q=4','q=5','q=6','q=7'});
title('energy for the central flux')



