non-offical implementation of undeepvo

0. Prepare dataset
run generate_train.m with matlab 

1. Activate tensorflow
source ~/tensorflow/bin/activate 

2. Training
python undeepvo_main.py --mode train --model_name my_model --data_path /media/youngji/storagedevice/naver_data/kitti_odometry/dataset/ --filenames_file ./utils/filenames/kitti_train_files2.txt --log_directory ./tmp/

3. visualization
tensorboard --logdir=./tmp/my_model
 
4. Testing
python monodepth_main.py --mode test --data_path /media/youngji/storagedevice/naver_data/kitti_odometry/dataset/ \
--filenames_file ./utils/filenames/kitti_train_files2.txt --log_directory ./tmp/ \
--checkpoint_path ./tmp/my_model/model-181250

python undeepvo_test.py --checkpoint_path ./tmp/my_model3/model-30000
