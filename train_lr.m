function [w]=train_lr(X,t)
w_size=size(X,2)+1;
w=zeros(w_size,1); 
N=size(X,1);
y=ones(N,1);
bisedinput=[X y];
I = speye(w_size);
lambda = 0.5;
regularization = lambda * I;
Tbisedinput=bisedinput';
i = 1;
Divergence_Criteria = 10^-3;

while 1 == 1 
    y=sigmoid(bisedinput*w);
    R=y.*(1-y);
    Sparse_R=spdiags(R,0,N,N);
    z=(bisedinput*w)-(Sparse_R\(y-t));
    wold = w;
    w=((Tbisedinput*(Sparse_R*bisedinput)+regularization))\(Tbisedinput*(Sparse_R*z));
    if sumsqr(w - wold) <= Divergence_Criteria, break; end %For Divengence
    disp(i);
    i = i +1;
end

end