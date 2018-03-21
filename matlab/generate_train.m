%generating training txt file

save_fid = fopen('kitti_train_files.txt', 'wt');
for iter=1:4540
    fprintf(save_fid, 'sequences/00/image_2/%06d.png ',iter-1);
    fprintf(save_fid, 'sequences/00/image_3/%06d.png ',iter-1);
    fprintf(save_fid, 'sequences/00/image_2/%06d.png ',iter);
    fprintf(save_fid, 'sequences/00/image_3/%06d.png ',iter);
    fprintf(save_fid, '\n');
end

fclose(save_fid);