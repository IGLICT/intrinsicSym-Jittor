function my_rewrite_mesh(X,T,path_name, mesh_name1)
fileID = fopen([path_name mesh_name1 '.txt'],'w');
fprintf(fileID,"%d %d\n",size(X,1),size(T,1));
fprintf(fileID,'%f %f %f\n',X');
fprintf(fileID,'%d %d %d\n',(T-1)');
fclose(fileID);
end