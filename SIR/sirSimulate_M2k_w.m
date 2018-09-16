function [y Y Vhat] = sirSimulate_M2k_w(S0, I0, n)
% SIR generation for lhs design points
% last update: 9/2/2018 by WW

M = 2000;
beta = 0.5;
gamma = 0.5;
imm = 0;

k = size(S0,1); % number of design points
for m = 1:k
    y(m).n = zeros([1  n(m)]);
end

for m = 1:k
    for j = 1:n(m)
    curS = repelem(S0(m), M);
    curI = repelem(I0(m), M);
    curT = repelem(0, M);
    curS(1) = S0(m);
    curI(1) = I0(m);
    curT(1) = 0;
    count = 1;
    maxI = I0(m);
    while ((curI(count) > 0 && imm == 0) || (curT(count) < 100 && imm > 0)) 
        infRate = beta * curS(count)/(M) * (curI(count) + imm);
        recRate = gamma * curI(count);
        infTime = -1/infRate * log(rand(1));
        recTime = -1/recRate * log(rand(1));
        if (infTime < recTime) 
            curS(count + 1) = curS(count) - 1;
            curI(count + 1) = curI(count) + 1;
            maxI = max(maxI, curI(count + 1));
        else 
            curI(count + 1) = curI(count) - 1;
            curS(count + 1) = curS(count);
        end
        curT(count + 1) = curT(count) + min(infTime, recTime);
        count = count + 1;
    end
    maxI = maxI;
    totT = curT(count);
    totI = S0(m) - curS(count);
    S = curS;
    I = curI;
    R = M - curS - curI;
    T = curT;
    y(m).n(j) = totI/800;
    end
end

Y = zeros(k,1);
Vhat = zeros(k,1);

for m = 1:k
    Y(m) = mean(y(m).n);
    Vhat(m) = var(y(m).n);
end

end

    