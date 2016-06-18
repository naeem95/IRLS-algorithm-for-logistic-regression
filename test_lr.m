function [accuracy]=test_lr(w,testX,testt)
x_size=size(testX,1);
y=ones(x_size,1);
biasd_test_points=[testX y];
    posterior=sigmoid(biasd_test_points*w);
for i=1:x_size
    if(posterior(i)>(1-posterior(i)))
        y(i)=1;
    else
        y(i)=0;
    end
end
count=0;
for j=1:x_size
    if(y(j)== testt(j))
        count=count+1;
    end
end
accuracy=(count/x_size)*100;
end