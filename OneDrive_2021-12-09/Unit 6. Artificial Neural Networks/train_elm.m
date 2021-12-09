function perf=train_elm(x,y,h)
A=2*rand(h,size(x,2))-1;
H=radbas(A*x');
B=pinv(H')*y;
z=(H'*B)';
perf=sqrt(mse(y'-z));
end
