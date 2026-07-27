# Check GPU on Server
nvidia-smi

# Latest Mumax Binaries
https://mumax.github.io/download.html

# Get on server
wget https://mumax.ugent.be/mumax3-binaries/mumax3.12_linux_cuda12.9.tar.gz

# Extract archive 
tar -xvzf ...

# make executable
chmod +x mumax3

# Adding to path
export PATH="/workspace/mumax3.12_linux_cuda12.9:$PATH"
echo 'export PATH="/workspace/mumax3.12_linux_cuda12.9:$PATH"' >> ~/.bashrc

cp mumax3 /usr/local/bin

# Applying changes
source ~/.bashrc

# install python packages

pip install ubermag


# exectute script 

nohup python main.py & 
(output will automatically saved in nohup.log) 

watch -n 1 nvidia-smi

# download files

ssh 8r0ja8kgy438n0-64411cc3@ssh.runpod.io -i ~/.ssh/id_ed25519
ssh root@213.192.2.124 -p 40066 -i ~/.ssh/id_ed25519

scp -r -P 40066 -i ~/.ssh/id_ed25519 root@213.192.2.124:~/direct_mx/mumax_grid_runs ./runpod_server
