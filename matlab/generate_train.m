%generating training txt file
% image_left, image_right, image_left_next, image_right, 
% left_focal_length, left_c0, left_c1, 
% right_focal_length, right_c0, right_c1
% base_line, width, height 

data_num = [4540,1100,4660, 800, 270, 2760, 1100, 1100, 4070];
image_dims = [376, 376, 376, 375, 370, 370, 370 ,370, 370;
              1241, 1241, 1241, 1242, 1226, 1226, 1226, 1226, 1226];

left_focal_length = zeros(1,numel(data_num));
right_focal_length = zeros(1,numel(data_num));
left_c0 = zeros(1,numel(data_num));
left_c1 = zeros(1,numel(data_num));
right_c0 = zeros(1,numel(data_num));
right_c1 = zeros(1,numel(data_num));
base_line = zeros(1,numel(data_num));
base_str = '/media/youngji/storagedevice/naver_data/kitti_odometry/dataset/sequences/';
for dnum=1:numel(data_num)
    %read camera param
    num_str = sprintf('%02d/calib.txt',dnum-1);
    read_str = strcat(base_str,num_str);
    read_fid = fopen(read_str,'r');
    lines = fscanf(read_fid, '%s %f %f %f %f %f %f %f %f %f %f %f %f\n',[15,5]);
    left_focal_length(dnum) = lines(4,3);
    right_focal_length(dnum) = lines(4,4);
    left_c0(dnum) = lines(6,3);
    left_c1(dnum) = lines(10,3);
    right_c0(dnum) = lines(6,4);
    right_c1(dnum) = lines(10,4);
    base_line(dnum) = -lines(7,2)/lines(4,2);
    fclose(read_fid);
end

save_fid = fopen('../utils/filenames/kitti_train_files2.txt', 'wt');
for dnum=1:numel(data_num)

    for iter=1:data_num(dnum)
        fprintf(save_fid, 'sequences/%02d/image_2/%06d.png ', dnum-1, iter-1);
        fprintf(save_fid, 'sequences/%02d/image_3/%06d.png ', dnum-1, iter-1);
        fprintf(save_fid, 'sequences/%02d/image_2/%06d.png ', dnum-1, iter);
        fprintf(save_fid, 'sequences/%02d/image_3/%06d.png ', dnum-1, iter);
        fprintf(save_fid, '%f %f %f ', left_focal_length(dnum), left_c0(dnum), left_c1(dnum));
        fprintf(save_fid, '%f %f %f ', right_focal_length(dnum), right_c0(dnum), right_c1(dnum));
        fprintf(save_fid, '%f %f %f ', base_line(dnum), image_dims(1,dnum), image_dims(2,dnum));
        fprintf(save_fid, '\n');
    end
end
fclose(save_fid);
