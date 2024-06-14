# chess-hackathon

## Let's train some Chess AI people!

Welcome, welcome, let's get you started right away.

## STEP 1

To proceed, ensure you have WireGuard downloaded and installed, and your VPN active. Refer to the `isc-demos` repo under "Section 1.1 Setting up the VPN" for instructions on how to get your VPN set up. https://github.com/StrongResearch/isc-demos

## STEP 2

Next you'll need to register as a new user on Control Plane (https://cp.strongcompute.ai/). Once inside, you'll need to click on the menu top-right and then click on "User Credentials". Now click on "ADD SSH KEY" at the bottom of the page.

Open a terminal on your computer. If you haven't already generated a public/private key pair you will need to do so now with the following.

```bash
cd ~
ssh-keygen
```

Press ENTER at each prompt, do not change the name or location of the file. Eventually you should be shown some random ASCII art and have your terminal returned. Then run the following.

```bash
cd ~
cat .ssh/id_rsa.pub
```

Copy the public key text displayed in your terminal and paste it into the input field on Control Plane, then click "SUBMIT NEW SSH PUBLIC KEY".

## STEP 3

Next generate a new container for yourself by clicking "GENERATE" next to the appropriate UserOrg. If you only have one, all the simpler. You may need to click "GENERATE" a couple of times. Once your container is generated click "START" to start it. Once the container is started you will be shown an SSH command that you can use to enter your container and develop your project.

**It is strongly recommended to use VSCode for this demo! Please just download VSCode and use it from this point forward**.

A couple of things we'll want to install for VSCode before we proceed:
 - Remote - SSH
 - Jupyter

Click on "Extensions" on the left-side of the VSCode window, search for and install both of the above. You should then be able to click on "Remote Explorer" on the left-side of the VSCode window, then open the "REMOTES" accordion on the left side panel, hover over "SSH" with your cursor and click on the little cog icon and press ENTER. Then copy and pase the following into the file that opens which should be called `config`.

```bash
Host CHESSBOT
  HostName 192.168.127.70
  User root
  Port <PORT>
```

Replace `<PORT>` with the port number displayed within the SSH command shown next to your container in Control Plane, then save and close this file. Again on the "REMOTES" accordion title, hover over and you should be able to click the refresh icon. Then you should see `CHESSBOT` appear in the "SSH" list. Hover over `CHESSBOT` and click on the right-arrow that appears. Then click on "Open Directory" and open the default directory (just press ENTER). You may also need to elect to trust the authors of this folder. Please do so. 

**Note**: If you STOP and START your container, you will be assigned a new PORT number and you will need to update your config with the new PORT number.

Great we're now inside our running container in VSCode!

## STEP 5

Next you'll need to clone this repo.

```bash
cd ~
git clone https://github.com/StrongResearch/chess-hackathon.git
```

## STEP 6

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

## How to train our very own Chess AI

First choose a model! We have a directory `models` with a couple of options to choose from - a  `transformer` model and a `conv` (convolutional) model. By all means take a look at the model architecture, but the basics are very much in the names. One thing to note is that the model classes are both called `Model` which will be a requirement for our competitors. When you have decided which model you want to train, copy the corresponding python and yaml files from the `models` subdirectory into the main `demo-chess` repo directory and rename it `model.py` and `model_config.yaml` with the following.

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


...


## What it feels like to watch our AIs play chess

Next let's actually jump ahead and see what we're shooting for. Open up the jupyter notebook `gameplay.ipynb`, and step through cell-by-cell. You can run the cells with `shift + enter`.

You'll see we're initializing a pair of agents, each with `chg.Agent()`. When called without any `model` argument, these agents will just make random moves. When we add a model later, the moves will reflect the judgement of that model!




As you can see we can play repeated games to a maximum depth of 50 moves per agent, and after each game we can print out the points won by each agent in that game. A **win** is awarded 1 point, a **loss** is awarded -1 points, and **draw** is awarded 0 points.

Next you can inspect the results of the last game played. Find the line `move = 0`. When you run this cell, you will be shown details of the board state at that move (move 0 is the starting board state). Printed off you will see the move that was made in PGN notation, the board presented to the agent in numpy array format, and an SVG rendering of the board in that position. By changing `move = 0` to `move = 1, 2, etc.` you can step through the successive moves made by the agents in the game.

In the last cell of the notebook we can play successive `tournaments`. Each tournament is at least two games wherein each agent gets a turn at playing `white` and `black` for sake of fairness. If there is an overall winner from the first two games of the tournament, the tournament is over and the points awarded to each agent are displayed. If the overall outcome of the first two games is a draw, another two games are played, and so on until a winning agent can be declared.



