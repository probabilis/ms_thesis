function [matrix_3d, E, FoV,Ef] = readin(my_folder)
%READIN reads in all Tif-files from a folder and creates a 3 dimensional 
%       matrix
%Input: my_folder ... path to the folder with the desired Tif-files
%Output: matrix_3d ... 3 dimensional matrix of the k-space 2d-images

[E, FoV] = readin_E(my_folder);
[Ef] = readin_Ef(my_folder);
[matrix_3d] = readin_im(my_folder);

FoV=mean(FoV);

end

