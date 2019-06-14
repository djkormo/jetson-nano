# jetson-nano

## Installation before we begin to use the jetson nano board

### Based on


##### https://medium.com/@heldenkombinat/getting-started-with-the-jetson-nano-37af65a07aab

##### https://forums.fast.ai/t/share-your-work-here/27676/1274

##### https://blog.hackster.io/getting-started-with-the-nvidia-jetson-nano-developer-kit-43aa7c298797





#### Ready solution

--------------------------
Once it boots up and you’ve verified it’s on your network and the Internet:

    Go to the Network Settings and find the IP V4 address of your machine, write it down , or if you understand IP networking set up a fixed IP address .
    Use the PC terminal program to open an SSH session with your Jetson Nano.
    Use your file transfer program to transfer the files in the zip File 84 to your Nano’s user home directory.
    From your PC ssh into the IP address in step 1 above.
    From either the console or via an SSH connection, set execute permissions on the scripts you’ve just downloaded:

chmod +x fastai_jetson.sh
chmod +x jetson_headless.sh
chmod +x setup_swapfile.sh
chmod +x setup_jupyter.sh

Set up a Swap File:

The Nano has only 4GB of RAM (which the GPU shares), you’re going to need to setup up a swap file. If you already have one, skip this step. You can just run the setup_swapfile.sh from your terminal session:

./setup_swapfile.sh

Be sure to ONLY DO THIS ONCE, as it has nothing in the script to check if it was already setup. Verify you swap file is setup by doing:

free

you should see an 8GB swap file created

Install pytorch and fast.ai:

If at this point you want to try the standard fast.ai and pytorch install, go right ahead, it will fail. For a bunch of reasons I’m not going to go into now, the standard pip commands simply won’t work for this. But if you just run the fastai_jetson.sh script you downloaded it will install both. Now this will take a couple of hours at best, so don’t hold your breath.

./fastai_jetson.sh

Install jupyter notebook:

After fast.ai is installed, it tells you:
Done with part1 – now logout, login again and run setup_jupyter.sh

This is because the jupyter install doesn’t export the shell variables it need to run. So shutdown all your terminals, SSH sessions etc. and just reboot the Nano from the GUI. Once it comes back up. Open up a terminal from the GUI and :

    Make sure that the jupyter_notebook_config.py file you downloaded is in the nano’s home directory.
    run ./setup_jupyter.sh

./setup_jupyter.sh

This also takes a while, so again don’t hold your breath . The last step of this script asks for your jupyter password. This IS NOT your login password, this is a separate password you can use to log into jupyter notebook from any PC on your network, so pick an appropriate password and write it down. The default jupyter notebook install only lets you log in from the console or GUI, the modified jupyter_notebook_config.py file you downloaded and the script installs allows you to login from any machine on your network. To run jupyter notebook you will have to open a terminal or ssh instance and run:

jupyter notebook

If it doesn’t run, it’s probably because you didn’t log out and in again.
That’s it. Your done, You can now run pytorch and fast.ai. But if you’re like me, you don’t need a GUI on the nano, and want all the memory you can get to run programs.

A Note about Python, Pip and VirtualEnv:

Some experienced python users are used to a virtual environment (virtualenv, conda) which requires you to activate it ‘source activate’ before you install or run software you’ve installed in that environment. We haven’t installed that (Yes we probably should have), one of the side effects of this is that the pip and python commands will run python3 or pip3 automatically, if that’s the active environment.
You must use pip3 and python3 to run pip and python respectively. So if you’re running some notebook that uses !python xyz, it won’t work unless you change it’s code to !python3 xyz.

Memory isn’t everything, but it’s definitely something:

Back in the old days (of say 2010), 4GB was a lot of memory. And If you’re not using the GPU on this board, it is enough to get your notebooks running well (the 8 GB of swap file helps quite a bit). But if you’re using CUDA, it doesn’t run on the swap disk, so you need each and every byte of that 4GB. To get that, it’s time to jettison the GUI and run via a remote console using SSH. Running the jetson_headless.sh script will uninstall the GUI, and purge a couple of unnecessary packages that take up over 300MB of RAM. So after you run this and reboot, you’ll only have console access to the Nano, but you’re the machine will start using only about 378MB of RAM, leaving you with 3.6GB for pytorch and fast.ai.

1.run:
./jetson_headless.sh
2. reboot and ssh into your nano.

A Note about changes:

As of April 2019, this hacky install method works and installs the latest versions of both pytorch 1.0 and fast.ai 1.0, but things change. In the future you will have to update one or more packages or fast.ai itself. Hopefully some clever soul will figure out how to do that and maybe even build a git repo. My work here is done.

-------------------------