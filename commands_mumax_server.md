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
export PATH="/path/to/extracted/mumax3:$PATH"

# Applying changes
source ~/.bashrc

# install python packages

pip install ubermag


# exectute script 

nohup python main.py & 
(output will automatically saved in nohup.log) 

watch -n 1 nvidia-smi

# download files

ssh root@213.173.110.147 -p 17326 -i ~/.ssh/id_ed25519

scp -r -P 17326 -i ~/.ssh/id_ed25519 root@213.173.110.147:/workspace/software ./runpod_server
