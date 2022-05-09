clear
addpath(genpath(pwd));
obj_path    = '..\data\rawdata\';
output_path = '..\data\annotated\';

if ~exist(output_path,'file')
    mkdir(output_path)
end

objlist=dir([obj_path,'*.obj']);

for i=1:length(objlist)
    obj_name=[obj_path,objlist(i).name];
    mat_name=[obj_path,objlist(i).name(1:end-4),'_evecs.mat'];
    new_mat_name=[output_path,objlist(i).name(1:end-4),'_anno.mat'];
    disp(new_mat_name);
    if exist(new_mat_name,'file')
        continue;
    end
    load(mat_name);
    [X,T] = readOBJfast(obj_name);
    model_sign = mv_render_choose(X,T,model_evecs);
    save(new_mat_name,'model_sign','model_evecs');
end

