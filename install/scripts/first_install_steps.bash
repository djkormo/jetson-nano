# poczatek

sudo apt-get update
sudo apt-get upgrade

# restart systemu

sudo reboot 


# narzedzia do monitoringu GPU 
https://github.com/rbonghi/jetson_stats


# instalacja pythona i kompilatorow

sudo apt-get install git cmake
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install libhdf5-serial-dev hdf5-tools
sudo apt-get install python3-dev
sudo apt-get install python3-matplotlib
sudo apt-get install libfreetype6-dev # for matplot 
sudo apt-get install python3-pil # for matplot 

# instalacja pip

wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
rm get-pip.py

# instalacja srodowiska wirtualnego dla pythona

sudo pip install virtualenv virtualenvwrapper


# edycja pliku
nano ~/.bashrc

# i dodanie na jego koncu

# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh

# tworzenie srodowiska wirtualnego o nazwie deep_learning  dla pythona 3
mkvirtualenv deep_learning -p python3

# uruchomienie kontekstu srodowiska
workon deep_learning

pip install numpy

pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu==1.13.1+nv19.3

pip install scipy
pip install keras
pip install dlib
pip install imutils

# instalacja notatnikow jupytera
pip install ipykernel
pip install jupyter notebook 
ipython kernel install --user --name=deep_learning


# uruchomienie notatnika
# naprawa uprawnien
sudo chown -R djkormo:djkormo ~/.local/share/jupyter 

jupyter notebook




