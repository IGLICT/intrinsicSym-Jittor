function mesh=find_mesh_lbo_newa(file_name,num_ev)
%     [X, T] = readOff([file_name '.off']);
    obj = readObj([file_name '.obj']);
    X=obj.v;
    T=obj.f.v;
    mesh.vertices = X;
    mesh.triangles = T;
    mesh.nv = length(mesh.vertices);
    mesh.nf = length(mesh.triangles);
    mesh.ta = face_areas( mesh );
    mesh.va = vertex_areas( mesh );
    [mesh.VA,mesh.IVA,mesh.TA,mesh.ITA] = mass_matrices( mesh );
    mesh.Nf = face_normals( mesh );
    [mesh.E1,mesh.E2,mesh.E3] = face_edges( mesh );
    mesh.G = grad( mesh );
    mesh.D = div( mesh );
    [mesh.laplaceBasis, mesh.BI, mesh.eigenvalues ] = func_basis( mesh, num_ev);
    mesh.eigenvalues=diag(mesh.eigenvalues);
    mesh.areaWeights=mesh.va;
end
function [ nv ] = normv( v )
    nv = sqrt(sum(v.^2,2));
end
       
function [ nf ] = normalize_f( f )
    nf = (f - min(f)) / (max(f)-min(f));
end

function [ snv ] = snormv( v )
    snv = sum(v.^2,2);
end
function [ rvf ] = rotate_vf( mesh, vf )
    vf = reshape(vf,mesh.nf,3);
    rvf = cross( mesh.Nf, vf );
end

function [ nnv ] = normalize_vf( v )
    vn = normv(v); I = vn > 0;
    nnv = v;
    nnv(I,:) = v(I,:) ./ repmat(vn(I),1,3);
end
function [ ta ] = face_areas( mesh )
X = mesh.vertices;
T = mesh.triangles;

P1 = X(T(:,1),:) - X(T(:,2),:);
P2 = X(T(:,1),:) - X(T(:,3),:);

ta = normv( cross( P1, P2 ) ) / 2;
end

function [ va ] = vertex_areas( mesh )
va = full( sum( mass_matrix(mesh), 2 ));
end

function [ M ] = mass_matrix( mesh )
T = double( mesh.triangles );

I = [T(:,1);T(:,2);T(:,3)];
J = [T(:,2);T(:,3);T(:,1)];
Mij = 1/12*[mesh.ta; mesh.ta; mesh.ta];
Mji = Mij;
Mii = 1/6*[mesh.ta; mesh.ta; mesh.ta];
In = [I;J;I];
Jn = [J;I;I];
Mn = [Mij;Mji;Mii];
M = sparse(In,Jn,Mn,mesh.nv,mesh.nv);
end

function [ VA, IVA, TA, ITA] = mass_matrices( mesh )
sv = mesh.nv; sf = mesh.nf;
VA = spdiags(mesh.va,0,sv,sv);
IVA = spdiags(1./mesh.va,0,sv,sv);
TA = spdiags([mesh.ta; mesh.ta; mesh.ta],0,3*sf,3*sf);
ITA = spdiags(1./[mesh.ta; mesh.ta; mesh.ta],0,3*sf,3*sf);
end

function [ Nf ] = face_normals( mesh )
X = mesh.vertices;
T = mesh.triangles;

P1 = X(T(:,1),:) - X(T(:,2),:);
P2 = X(T(:,1),:) - X(T(:,3),:);

Nf = cross( P1, P2 );
Nf = normalize_vf( Nf );
end

function [ E1, E2, E3 ] = face_edges( mesh )
X = mesh.vertices;
T = mesh.triangles;

E1 = X(T(:,3),:) - X(T(:,2),:);
E2 = X(T(:,1),:) - X(T(:,3),:);
E3 = X(T(:,2),:) - X(T(:,1),:);
E = [E1; E2; E3];

mesh.mel = mean( normv( E ) );
end


function [ G ] = grad( mesh )
% G corresponds to eq. (3.9) in Polygon mesh processing book
I = repmat(1:mesh.nf,3,1);
II = [I(:); I(:)+mesh.nf; I(:)+2*mesh.nf];

J = double( mesh.triangles' );
JJ = [J(:); J(:); J(:)];

RE1 = rotate_vf( mesh, mesh.E1 );
RE2 = rotate_vf( mesh, mesh.E2 );
RE3 = rotate_vf( mesh, mesh.E3 );

S = [RE1(:) RE2(:) RE3(:)]'; SS = S(:);

G = sparse(II,JJ,SS,3*mesh.nf,mesh.nv);
G = .5 * mesh.ITA * G;
end

function [ D ] = div( mesh )
% D corresponds to eq. (3.12) in Polygon mesh processing book
D = - mesh.IVA * mesh.G' * mesh.TA;
end
function [ B, BI, D ] = func_basis( M, k )
L = - M.D * M.G;
A = M.VA;
W = A*L;
tic; 
[ev,el] = eigs(W,A,k,'SM');
[~,ii] = sort(diag(el)); el = el(ii,ii); ev = ev(:,ii);
B = ev; BI = ev'*A; D = el;
end