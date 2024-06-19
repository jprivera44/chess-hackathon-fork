# chess-hackathon

## Let's train some Chess AI people!

Welcome, welcome, let's get you started right away.

## STEP 0

Before getting started there are a couple of pieces of software you will need to download and install; WireGuard, and VSCode.

 - WireGuard is our VPN software of choice: https://www.wireguard.com/install/
 - We will be using VSCode to access and develop in our Instant SuperComputer (ISC) environment: https://code.visualstudio.com/download

Once you have installed VSCode, also ensure you have the following VSCode extensions installed:
 - Remote - SSH
 - Jupyter

All good? Great let's go!

## STEP 1

Before connecting to the Strong Compute ISC, you must have recieved login credentials from Strong Compute by email. Please reach out to us if you have not recieved this email.

### For MacOS and Windows
1. Once you have WireGuard installed, visit the Strong Compute FireZone portal at the website advised in the login credentials email from us and 
    login with the credentials advised in that email.
2. In the FireZone portal, click "Add Device". You can give the configuration any name you like, if you do not provide a 
    name for the configuration, FireZone will automatically generate one for you.
3. Click on "Generate Configuration" and then "Download WireGuard Configuration" to download the tunnel config file. 
    Save the tunnel config file somewhere suitable on your machine.
4. Within the WireGuard application, select "Import Tunnel(s) from File" and select the tunnel config file from where 
    you saved it.
5. Ensure the VPN tunnel is enabled when accessing the Strong Compute ISC. When the VPN is correctly enabled you should 
    be able to open a terminal, run `ping 192.168.127.70` and recieve `64 bytes from 192.168.127.70`.

### For Linux
1. Once you have WireGuard installed, visit the Strong Compute FireZone portal at the website advised in the login credentials email from us and 
    login with the credentials advised in that email.
2. In the FireZone portal, click "Add Device". You can give the configuration any name you like, if you do not provide a 
    name for the configuration, FireZone will automatically generate one for you.
3. Click on "Generate Configuration" and then "Download WireGuard Configuration" to download the tunnel config file. 
    Save the tunnel config file somewhere suitable on your machine.
4. Within your terminal run the following commands (you will need sudo-level access):
 - Ensure the package manager is up-to-date with `sudo apt update`
 - Install WireGuard with `sudo apt install -y wireguard`
 - Open the WireGuard config file with `sudo nano /etc/wireguard/wg0.conf`
5. In the WireGuard config file, delete or comment out the line starting with "DNS" (you can comment out by adding a `#` 
    to the start of the line).
6. Open the tunnel config file you downloaded from FireZone portal, copy and paste the contents of the tunnel config 
    file into the WireGuard config file. Your WireGuard config file should look as follows.

```
# DNS = 1.1.1.1,10.10.10.1

[Interface]
PrivateKey = <private-key>
Address = <ip address>
MTU = 1280

[Peer]
PresharedKey = <preshared-key>
PublicKey = <public-key>
AllowedIPs = <ip address>
Endpoint = <endpoint>
PersistentKeepalive = 15
```

7. Save the updated WireGuard config file and close nano.
8. Ensure the VPN tunnel is enabled when accessing the Strong Compute ISC. You can enable the VPN with 
    `sudo wg-quick up wg0` and disable with `sudo wg-quick down wg0`. When the VPN is correctly enabled you should be 
    able to open a terminal, run `ping 192.168.127.70` and recieve `64 bytes from 192.168.127.70`.
   
## STEP 2

 - Next you'll need to register as a new user on Control Plane (https://cp.strongcompute.ai/).
 - Once inside, you'll need to click on the menu top-right and then click on "User Credentials".
 - Now click on "ADD SSH KEY" at the bottom of the page.
 - Open a terminal on your computer.
 - If you haven't already generated a public/private key pair you will need to do so now with the following.

```bash
cd ~
ssh-keygen
```

 - Press ENTER at each prompt, do not change the name or location of the file.
 - Eventually you should be shown some random ASCII art and have your terminal returned.
 - Then run the following.

```bash
cd ~
cat .ssh/id_rsa.pub
```

 - Copy the public key text displayed in your terminal and paste it into the input field on Control Plane.
 - Then click "SUBMIT NEW SSH PUBLIC KEY".

## STEP 3

 - Next generate a new container for yourself by clicking "GENERATE" next to the appropriate UserOrg. If you only have one, all the simpler.
 - You may need to click "GENERATE" a couple of times.
 - Once your container is generated click "START" to start it.
 - Once the container is started you will be shown an SSH command that you can use to enter your container and develop your project.
 - Open VSCode
 - Having installed the **Remote - SSH** extension (see above) you should then be able to click on "Remote Explorer" on the left-side of the VSCode window
 - Then open the "REMOTES" accordion on the left side panel, hover over "SSH" with your cursor and click on the little cog icon and press ENTER.
 - Then copy and pase the following into the file that opens which should be called `config`.

```bash
Host CHESSBOT
  HostName 192.168.127.70
  User root
  Port <PORT>
```

 - Replace `<PORT>` with the port number displayed within the SSH command shown next to your container in Control Plane, then save and close this file.
 - Again on the "REMOTES" accordion title, hover over and you should be able to click the refresh icon.
 - Then you should see `CHESSBOT` appear in the "SSH" list.
 - Hover over `CHESSBOT` and click on the right-arrow that appears.
 - Then click on "Open Directory" and open the default directory (just press ENTER).
 - You may also need to elect to trust the authors of this folder. Please do so. 

**Note**: If you STOP and START your container, you will be assigned a new PORT number and you will need to update your config with the new PORT number.

Great we're now inside our running container in VSCode!

## STEP 4

Next you'll need to clone this repo. Drop the commands below into the terminal in VSCode.

```bash
cd ~
git clone https://github.com/StrongResearch/chess-hackathon.git
```

## STEP 5

Jolly good. Moving on, create and source a new virtual environment, let's call it `~/.chess`. 

```bash
cd ~
python3 -m virtualenv ~/.chess
source ~/.chess/bin/activate
```

Now install the `chess-hackathon` repo dependencies with the following command.

```bash
cd ~/chess-hackathon
pip install -r requirements.txt
```

Ok we're all set up to start training AI models!

## Train our very own Chess AI

First choose a model! We have a directory `models` with a couple of options to choose from - a  `transformer` model and a `convolutional` model. By all means take a look at the model architecture, but the basics are very much in the names. One thing to note is that the model classes are both called `Model` which will be a requirement for our competitors. When you have decided which model you want to train, copy the corresponding python and yaml files from the `models` subdirectory into the main `demo-chess` repo directory and rename it `model.py` and `model_config.yaml` with the following.

```bash
cp ~/chess-hackathon/models/<your_model.py> ~/chess-hackathon/model.py
cp ~/chess-hackathon/models/<your_model.yaml> ~/chess-hackathon/model_config.yaml
```

We have just simulated one essential step for our competitors - they must generate and submit a `model.py` file and a `model_config.yaml` file at the end of their hacking!

### Get set up on Control Plane
A couple more things we will need to get set up on Control Plane.

Visit Control Plane and click on "Project". Click on "NEW PROJECT", give your new project a name, and click "CREATE". 

Next click on "Datasets". Again we expect that by game day we will have the `chess` dataset made publically available for all users, but for now we're going to need to manually set up access for everyone to the chess dataset. Kind of no way to do this until everyone has created their login etc. so we'll do this together in real time for the dry run. The dataset comprises board states (numpy arrays representing where pieces are on the board) and evaluations of those board states care of StockFish (integer values in the range of -10,000 to +10,000 where large positive numbers means a very good position and large negative numbers means a very bad position).

Finally we'll need to allocate everyone budget to spend. This will look something like everyone giving Adam their Organisation name, and Adam going into Strong Admin and crediting everyone with budget to spend.

### Launching our model to train
You can launch your model to train with `isc train run.isc`!

### Testing gameplay
While your model is training let's take a look at what the gameplay API looks like. Open up the notebook called `gameplay.ipynb`. You may be prompted to install jupyter things. Please do so. You may also need to click on the "Select Kernel" button at the top-right, find and select the `.chess` kernel (from the virtual environment we created).

Step through this notebook and see what we're working with!



