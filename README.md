# sonic2_reinforcement_learning
Deep Learning (RAF117F fall 2018)  
University of Iceland  
Final project for the course Deep Learning at University of Iceland.

## Example videos
Previous version after 6 days of training:  
[![youtube_v1](./gifs/v1.gif)](https://youtu.be/FdN4oRy5g6E)  

Couple of modifications were made to the environment and hyper-parameters were re-tuned:

• The agent is limited to seeing 3 frames out of 60 per second and would play the same
chosen action over the next 19 unseen frames.

• The environment terminated after 90 seconds instead of the default 10 minutes.  

Modified version after 48h of training:  
[![youtube_v2](gifs/v2.gif)](https://youtu.be/mKLSF36KtOY)  


## Prerequisites
Python 3  
numpy  
tensorflow  
Gym Retro  
Game ROM of Sonic the Hedghog 2  

## Running the code
Install gym retro, https://github.com/openai/gym  
```
sudo apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb
pip3 install gym
pip3 install gym retro
# pygame needed for some functions
pip3 install pygame
```
Install game rom from steam (if owned, else acquire the rom somehow else), https://store.steampowered.com/app/71163/Sonic_The_Hedgehog_2/
```
ls ~/.steam/steam/steamapps/common/'Sega Classics'/'uncompressed ROMs'/
```
Copy roms to some location with extention for system e.g. .md for mega drive, then import to gym retro
```
cp ~/.steam/steam/steamapps/common/'Sega Classics'/'uncompressed ROMs'/SONIC2_W.68K ~/roms/SONIC2_W.md
python3 -m retro.import ~/roms/

```
Run:
```
Python DQN_tensorflow.py
```

### Author
Eyþór Einarsson

### Links
* [RAF117F, University of Iceland](https://ugla.hi.is/kennsluskra/index.php?sid=&tab=nam&chapter=namskeid&id=70970220186)
* [UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
* [Reinforcement Learning, book by Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/the-book-2nd.html)
* [Andrew Ng's Deep Learning Specialization on coursera.com] {https://www.coursera.org/specializations/deep-learning?)

