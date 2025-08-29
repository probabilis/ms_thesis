function [matrix_3d] = readin_im(my_folder)
%READIN_IM Summary of this function goes here
%   Detailed explanation goes here

%define pixel size
pixel=1024;

%read in tif.files
filePattern = fullfile(my_folder, '*.TIF'); 
theFiles = dir(filePattern);

imageArray=cell(1,length(theFiles));

matrix_3d=zeros(pixel,pixel,length(theFiles));

for k = 1:length(theFiles)
    
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(my_folder, baseFileName);
    %fprintf(1, 'Now reading %s\n', fullFileName);
    imageArray{k} = flipud(imread(fullFileName));
    %generate 3d matrix
    matrix_3d(:,:,k)=imageArray{k};
    
end
end

