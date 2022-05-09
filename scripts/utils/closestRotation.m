function R = closestRotation(M,alpha)
if nargin < 2
    alpha = 0;
end
[uu,ss,vv] = svd(M); 
dd = diag(ss);

dd(dd > alpha) = 1;
ss = diag(dd);

R = uu*ss*vv'; 