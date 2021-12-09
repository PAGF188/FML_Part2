clear;x=load('hepatitis.data');
addpath("nnet/inst")
y=x(:,1);x(:,1)=[];h=1:10:200;nh=numel(h);rmse=zeros(1,nh);rmse(:)=0;
for i=1:nh
	rmse(i)=train_elm(x,y,h(i));
end
clf;plot(h,rmse,'bs-','LineWidth',2);
xlabel('hidden nodes');ylabel('RMSE')
title('hepatitis dataset');grid
