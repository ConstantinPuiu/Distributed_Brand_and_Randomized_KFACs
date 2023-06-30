1) untar imagenet_prep_sh_files.tar.xz 

############## Train Shell Script ############
2) in the most popular imagenet data format nowadays, all that is needed is to untar each class tar found in dir /imagenet/train/ . Written a shell for convenience. To be ran in imagenet/train/...

############## Val Shell Script ############
3) The shell script file imagenet_valprep.sh must be ran in the /.../imagenet/val/ directory if after untaring the data structure is not of the form .../imagenet/val/class_folder/....JPEG .
Running the .sh file will create the correct class folder structure for val

.sh file taken from https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh , all credits for file go to soumith, github repo: https://github.com/soumith/imagenetloader.torch

May also choose to run the command wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash . Put the shell file here for convenience.

