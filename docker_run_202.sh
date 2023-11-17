docker run --rm --name "sasaki_mau_0" --gpus '"device=0"' --shm-size 4096mb -v /home/sasaki/MAU:/home/MAU -v /mnt/hdd1/sasaki/MAU_data/4_datasets:/home/MAU/data -it sasaki/mau:latest
