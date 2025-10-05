function [Ef] = readin_Ef(myFolder)
%READIN reads in DAT.-files from folder and creates a matrix
% Input: my_folder ... path to the folder with the desired DAT-file
% Output: data ... matrix with E-Ef (first column) and counts (second col)

filePattern = fullfile(myFolder, '*.DAT'); 
theFiles = dir(filePattern);
baseFileName = theFiles.name;
fullFileName = fullfile(myFolder, baseFileName);
FileID = fopen(fullFileName);
datacell= textscan(FileID, '%s %f',...
    'delimiter','=', 'HeaderLines',21,'Collect',1,'CommentStyle','""');
fclose(FileID);
Ef=datacell{1,2};
% data_1=datacell{1,1};
% Ef=data_1(:,1:2);

% baseFileName = theFiles.name;
% fullFileName = fullfile(myFolder, baseFileName);
% FileID = fopen(fullFileName);
% datacell = textscan(FileID, '%f %f %f %f %f %f %f %f %f %f %f %q',...
%     'delimiter','\n', 'HeaderLines',90,'CollectOutput',1,'CommentStyle','""');
% fclose(FileID);
% data_1=datacell{1,1};
% data=data_1(:,1);




end
