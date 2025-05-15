rcp=matrix_3d_2;%
lcp=matrix_3d_1;%

r_ms=rcp(:,:,1)>250;
rcp=rcp.*r_ms;
lcp=lcp.*r_ms;

[X,Y]=meshgrid(-511:512);
FoV_1=38;
X=FoV_1./1024.*X;
Y=FoV_1./1024.*Y;

X_1=X(240:784,140:884);
Y_1=Y(240:784,140:884);

% flat field correction
f=fspecial('average',250);

ff_rcp=imfilter(rcp,f);
ff_lcp=imfilter(lcp,f);

rcp_2=rcp./ff_rcp;
lcp_2=lcp./ff_lcp;

MCD=(rcp_2-lcp_2)./(rcp_2+lcp_2);

%%
figure
colormap(gray)
surf(X,Y,MCD(:,:,1),'edgecolor', 'none')view(0,90)
set(gca,'Visible','off');
clim([-0.05 0.05])
axis equal
