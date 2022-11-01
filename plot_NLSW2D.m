% plot
uerrors = zeros(4,4);
uerrors_imag = zeros(4,4);
verrors = zeros(4,4);
verrors_imag = zeros(4,4);
meshwidth = zeros(4,4);

for q=3
    for i=1:4
        N = 8*(2^(i-1));
        [xloc,yloc,erru,erru_imag,errv,errv_imag] = NLSW2Dcentral_ssprk54(q,q,N,N,1);
        
        uerrors(q+1,i) = sqrt(erru);
        uerrors_imag(q+1,i) = sqrt(erru_imag);
        verrors(q+1,i) = sqrt(errv);
        verrors_imag(q+1,i) = sqrt(errv_imag);
        meshwidth(q+1,i) = 2*pi/N;
        
        if not(i==1)
		    r = -log(uerrors(q+1,i)/uerrors(q+1,i-1))/log(2);
		    fprintf('%d u %d %d real order = %4.3e \n', q, N, N/2, r);

		    r = -log(uerrors_imag(q+1,i)/uerrors_imag(q+1,i-1))/log(2);
		    fprintf('%d u %d %d imag order = %4.3e \n', q, N, N/2, r);

            r = -log(verrors(q+1,i)/verrors(q+1,i-1))/log(2);
            fprintf('%d v %d %d real order = %4.3e \n', q, N, N/2, r);

            r = -log(verrors_imag(q+1,i)/verrors_imag(q+1,i-1))/log(2);
            fprintf('%d v %d %d imag order = %4.3e \n', q, N, N/2, r);
        end
    end
end

