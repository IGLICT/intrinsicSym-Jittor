clear
addpath(genpath(pwd));

original_path = '../data/rawdata/';
output_path=original_path;
num_basis = 17; %do not need to plus one here

file_infos = dir(fullfile(original_path,'*.obj'));
file_names = {file_infos.name};
num_files = size(file_names, 2);

for i=1:num_files
    fprintf('computing %d\n',i);
    file_name =  char(file_names(1,i));
%     compute_ss(fullfile(original_path, file_name), fullfile(output_path, [file_name(1:end-4), '.mat']));
    compute_evecs(original_path,file_name(1:end-4), fullfile(output_path, [file_name(1:end-4), '_evecs.mat']), num_basis);
end