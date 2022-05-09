clear
addpath(genpath(pwd));
file_name = 'cone33';
mat_name='../data/predict/S.mat';
%evecs_name='../data/demo/scape_mesh071_anno.mat';
evecs_name=['../data/demo/',file_name,'_evecs.mat'];
load(mat_name);
load(evecs_name);
num_eig_basis=16;%size(ss_our,2);
num_basis=num_eig_basis+1;
our_ss=ss_our(:,1:num_eig_basis)*2-1;

if size(our_ss,2)~=num_eig_basis
   error('Size does not match'); 
end
num_step=20;

%mesh_name1='../data/demo/scape_mesh071';
mesh_name1=['../data/demo/',file_name];
mesh=find_mesh_lbo_newa(mesh_name1,num_basis);
X=mesh.vertices;
T=mesh.triangles;
ss=our_ss;
basis_A=model_evecs(:,1:num_eig_basis);
C=diag(ss);
numIter = 8;
err=[];
C = closestRotation(C);
 for k = 1:numIter
    CP=C*basis_A';
    nnidx = annquery(basis_A',CP, 1);
    W = basis_A(nnidx,:)'*basis_A(:,:);
    [uu, ~, vv] = svd(W);
    C = uu*vv';
    err=[err;norm(basis_A(nnidx,:)'-C*basis_A','fro')];
 end
CP=C*basis_A';

nnidx = annquery(basis_A',CP,1);
pts_set=randperm(size(X,1),300);
figure('units','normalized','outerposition',[0 0 1 1])
colormap(repmat([255 255 255]/255,mesh.nv,1));
h=trisurf(T,X(:,1),X(:,2),X(:,3),1:mesh.nv);hold on;
axis off;
axis image;
view(-45, 0)
shading interp
camlight(0,0)
h.FaceLighting = 'phong';
h.AmbientStrength = 0.3;
h.DiffuseStrength = 0.75;
h.SpecularStrength = 1;
h.SpecularExponent = 79;
for i=1:length(pts_set)
    if (abs(X(pts_set(i),2) -X(nnidx(pts_set(i)),2))>0.1)
        continue;
    end
    if (abs(X(pts_set(i),1) -X(nnidx(pts_set(i)),1))>0.05)
        continue;
    end
    if (abs(X(pts_set(i),3) +X(nnidx(pts_set(i)),3))>0.05)
        continue;
    end
    plot3([X(pts_set(i),1) X(nnidx(pts_set(i)),1)], [X(pts_set(i),2) X(nnidx(pts_set(i)),2)], [X(pts_set(i),3) X(nnidx(pts_set(i)),3)], 'Color', [39	64	139]/255, 'LineWidth', 1);
end

