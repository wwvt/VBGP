function [y, Y, Vhat] = SsInventoryData_w(Korder,pbklog,meanD,c,h,T,n,Xdesign,CRN)
% Created on 7/5/2011
% Updated on 1/28/2017

% This function is to generate the (S,s) inventory data for a period of length T: for both the 
% the long-run avg. inventory cost per period and the gradient esimate with
% respect to S and s at each pair of the design point (S,s)


% T: T-period simulation length 
% n: number of replications of T-period simulation at each design point
%    (S,s)
K = size(Xdesign,1);  %number of design points
delta = Xdesign(:,1);
s = Xdesign(:,2);
S = delta+s;

% CRN is not used in driving the simulation 
for l=1:K 
         y(l).n = zeros([1 n(l)]); 
% UInitialPos = rand;
          W=0;
           
          for j=1:n(l)
              Jsum = 0;
              haz = 0;
              n_star = 0;
              Hsum=0;
              Psum=0;
              
     % independently generated random demand of each period at each (S,s) 
              D = exprnd(meanD*ones(T,1),T,1);
              
              if j==1
%              W = UInitialPos*S(l);  % initial inventory position
              W = S(l);
              else
              W = Wlast;
              end
              
              z = S(l) - W; % initial order quantity
               
              
              % for each period
                for i = 1:T 
                                 
                     if W>0  % holding inventory
                         Jsum = Jsum +h*W;
                         Hsum = Hsum +h;
                     else    % backlogged  
                         Jsum = Jsum -pbklog*W;
                         Psum = Psum -pbklog;
                     end
              
                       if W < s(l)  % fall below reorder point
                          n_star = n_star+1;
                          haz = haz+ exppdf(z,meanD)/(1-expcdf(z,meanD));
                          Jsum = Jsum+Korder+c*(S(l)-W);
                          z = S(l)-s(l); 
                          W = S(l)-D(i);
                       else
                          z = W-s(l);
                          W = W-D(i);                   
                       end
              
                 end

               y(l).n(j) = Jsum/T;
               Wlast = W;
          end %(end of loops of simulation reps)
 %(end of loops of design points) 
%Y = y(l).n;
    Y(l)  = mean(y(l).n); 
    Vhat(l) = var(y(l).n,0,2);
  end 
