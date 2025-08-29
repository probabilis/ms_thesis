function [E, FoV] = readin_E(myFolder)
%READIN reads in DAT.-files from folder and creates a matrix
% Input: my_folder ... path to the folder with the desired DAT-file
% Output: data ... matrix with E-Ef (first column) and counts (second col)

filePattern = fullfile(myFolder, '*.DAT'); 
theFiles = dir(filePattern);

cellArray=cell(1,length(theFiles));
E=[];
FoV=[];

for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(myFolder, baseFileName);
    FileID = fopen(fullFileName);
    cellArray{k} = textscan(FileID, '%f %f %f %f %f %f %f %f %f %f %f %q',...
    'delimiter','\n', 'HeaderLines',90,'CollectOutput',1,'CommentStyle','""');
    fclose(FileID);
    E=[E; cellArray{1,k}{1,1}(:,1)];
    FoV=[FoV; cellArray{1,k}{1,1}(:,9)];
end



% baseFileName = theFiles.name;
% fullFileName = fullfile(myFolder, baseFileName);
% FileID = fopen(fullFileName);
% datacell = textscan(FileID, '%f %f %f %f %f %f %f %f %f %f %f %q',...
%     'delimiter','\n', 'HeaderLines',90,'CollectOutput',1,'CommentStyle','""');
% fclose(FileID);
% data_1=datacell{1,1};
% data=data_1(:,1);




end
