docker run --rm --name "sasaki_mau_0" --gpus '"device=0"' --shm-size 4096mb -v /home/sasaki/MAU:/home/MAU -v /mnt/hdd2/sasaki/MAU_data:/home/MAU/data -it sasaki/mau:latest
