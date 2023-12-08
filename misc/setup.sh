# bash script for setting up workstation
# run this immediately after installing ubuntu and GPU drivers (Software & Updates > additional drivers)


# update system
sudo apt update
sudo apt upgrade

# install and set up ssh
sudo apt install ssh
ssh-keygen -t ed25519 -C "david"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# python
sudo apt install wget build-essential libncursesw5-dev libssl-dev \
libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7 python3.8 python3.9 python3.10 python3.11 python3.12

# git
sudo apt install git
git config --global user.email "ddcoggan@icloud.com"
git config --global user.name "ddcoggan"

# github setup
sudo apt install curl
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y
gh auth login

# install gcc and cuda (see instructions at https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
# these are the instructions as of 23-06-01
sudo apt-get install linux-headers-$(uname -r)
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
reboot

