function compute_evecs(path_name, shape_name, save_filename, num_eig_basis)
num_basis=num_eig_basis+1;
mesh_name1=[path_name, shape_name];
disp(mesh_name1);
mesh=find_mesh_lbo_newa_obj(mesh_name1,num_basis);
X=mesh.vertices;
model_evecs=mesh.laplaceBasis(:,2:end);
save(save_filename,'model_evecs');
end