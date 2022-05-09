load('/home/shuzhi/work_files/research/symmetry/our_results/test_set_ss/davidS.mat')
reshaped_ss = reshape(ss_our', 20, []);
reshaped_gt = reshape(gt_our', 20, []);

batch_i = 2;
mesh_path = '/home/shuzhi/work_files/research/symmetry/previous_code/IntSym/toscahires-off/';

shape_path = name(batch_i,:);
name_idx = find(shape_path=='/',1,'last')+1;
shape_name = shape_path(name_idx:end-4);
shape_name = shape_name(1:end-7);

num_eig_basis=size(reshaped_ss, 1);
our_ss = double(reshaped_ss(:, batch_i)');
gt_ss = double(reshaped_gt(:, batch_i)');
our_ss = (our_ss-0.5)*2;
gt_ss = (gt_ss-0.5)*2;

use_eccv_to_visualize_ss(mesh_path, shape_name, our_ss, num_eig_basis)