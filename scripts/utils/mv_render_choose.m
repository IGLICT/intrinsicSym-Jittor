function [labels] = mv_render_choose(V,F,eigvector)

    %figure();
    opts.az = [0:60:300];
    opts.el = 30;
    opts.use_dodecahedron_views = false;
    opts.colorMode = 'rgb';
    opts.outputSize = 224;
    opts.minMargin = 0.1;
    opts.maxArea = 0.3;
    opts.figHandle = [];
    opts.figHandle = figure;
    ims = cell(1,length(opts.az));
    num_pic=17;
    
    redothis = 0;
    
    labels = zeros(1,num_pic);
    
    start = 1;
    %figure(4)
    

    
    j=1;
    az=opts.az(j);
    el=opts.el;
    i = start  ;  
    while i < start+num_pic-1
        
        cofunc = eigvector(:,i);
        Coeff = eigvector' * (cofunc - repmat(mean(cofunc),size(V,1),1));
        orginalColor = eigvector * Coeff;

    %     if ~exist(save_path,'dir')
    %         mkdir(save_path);
    %     end
        
        h = trimesh(F, V(:,1),V(:,2),V(:,3), orginalColor, 'FaceColor', 'interp', 'EdgeColor', 'none', ...
            'AmbientStrength', 0.3, 'DiffuseStrength', 0.6, 'SpecularStrength', 0.0, 'FaceLighting', 'flat');
        set(gcf, 'Color', 'w', 'Renderer', 'OpenGL');
        set(gca, 'Projection', 'perspective');    
        axis equal;
        axis off;
        view(az,el);
        title('PRESS 1: pos. 2: neither. 3: neg. 4, 5 :change view. 6: redo');

        button = 1;
        while button<49 || button >51
            [px, py, button] = ginput(1);
            switch button
                case 49
                    label = 1;
                case 50
                    label = 0;
                case 51
                    label = -1;
                case 52
                    el = el + 20;
                    if el>360
                        el = el - 360;
                    end
                    view(az,el);
                    continue
                case 53
                    az = az + 20;
                    if az>=360
                        az = az - 360;
                    end
                    view(az,el);
                    continue
                case 54
                    redothis = 1;
                    break
            end
            labels(i) = label;
        end
        if redothis == 1
            redothis = 0;
            i = start - 1;
            
        end
        i = i + 1;
    end
   

end

function keypressfcn(h,evt)
    fprintf('Press\n');
    global key_idx;
    key_idx = evt.key;
end

function im = resize_im(im,outputSize,minMargin,maxArea)

max_len = outputSize * (1-minMargin);
max_area = outputSize^2 * maxArea;

nCh = size(im,3);
mask = ~im2bw(im,1-1e-10);
mask = imfill(mask,'holes');
% blank image (all white) is outputed if not object is observed
if isempty(find(mask, 1)),
    im = uint8(255*ones(outputSize,outputSize,nCh));
    return;
end
[ys,xs] = ind2sub(size(mask),find(mask));
y_min = min(ys); y_max = max(ys); h = y_max - y_min + 1;
x_min = min(xs); x_max = max(xs); w = x_max - x_min + 1;
scale = min(max_len/max(h,w), sqrt(max_area/sum(mask(:))));
patch = imresize(im(y_min:y_max,x_min:x_max,:),scale);
[h,w,~] = size(patch);
im = uint8(255*ones(outputSize,outputSize,nCh));
loc_start = floor((outputSize-[h w])/2);
im(loc_start(1)+(0:h-1),loc_start(2)+(0:w-1),:) = patch;

end

