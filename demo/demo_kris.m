function [y, Y, Vhat] = demo_kris(X,n)
        
k = size(X,1); % number of design points
Y = zeros(k,1);
Vhat = zeros(k,1);

        for m = 1:k
                y(m).n = zeros([1 n(m)]);   
                y(m).n = normrnd(sin(X(m)),sqrt(0.05*X(m).*X(m)+0.01),[1 n(m)]);
                Y(m)  = mean(y(m).n); 
                Vhat(m) = var(y(m).n);
        end
end