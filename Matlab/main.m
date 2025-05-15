% % %main file
% % %read in image files
% % %
%clear all
% % %specify folder containing the TIF-files
%90°
my_folder_1='C:\Users\jaukt\Desktop\Magna-COCK\MCD size\Raw Data\2021-09-17\10_FoV38_E4c0_ap1750_Pharos91c5_Magneto4_210917_134820\10_FoV38_E4c0_ap1750_Pharos91c5_Magneto4_210917_134820\Sum';
% my_folder_2_up='C:\Users\jaukt\Desktop\Magna-COCK\Kspace\Pump-Probe\2021-02-08\3c7\90';
%0°
my_folder_2='C:\Users\jaukt\Desktop\Magna-COCK\MCD size\Raw Data\2021-09-17\11_FoV38_E3c9_ap1750_Pharos1c5_Magneto4_210917_140920\11_FoV38_E3c9_ap1750_Pharos1c5_Magneto4_210917_140921\Sum';
% my_folder_2_down='C:\Users\jaukt\Desktop\Magna-COCK\Kspace\Pump-Probe\2021-02-03\3c7\0';

% % % % %read in TIF 
[matrix_3d_1, E_1, FoV_1, Ef_1] = readin(my_folder_1);
[matrix_3d_2, E_4, FoV_2, Ef_2] = readin(my_folder_2);

% [matrix_3d_2, E_2, FoV_2, Ef_2] = readin(my_folder_2);
% [matrix_3d_2_down, E_2, FoV_2, Ef_2] = readin(my_folder_2_down);
 
 
% pe_30_90=reshape(sum(sum(matrix_3d_1(240:784,140:884,2:end))),size(E_1));
% pe_30_0=reshape(sum(sum(matrix_3d_2(240:784,140:884,2:end))),size(E_1));

