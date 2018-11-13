# sonic2_reinforcement_learning
## Deep Learning (RAF117F fall 2018)
## University of Iceland

Final project for the course Deep Learning at University of Iceland.

## Example videos
Previous version after 10h of training:  
[![Demo Doccou alpha](https://j.gifs.com/Yvj9pW.gif)](https://youtu.be/mKLSF36KtOY)

Couple of modifications were made to the environment and hyper-parameters were re-tuned:

• The agent is limited to seeing 3 frames out of 60 per second and would play the same
chosen action over the next 19 unseen frames.

• The environment terminated after 90 seconds instead of the default 10 minutes.
Modified version after 48h of training:  
[![Demo Doccou alpha](https://share.gifyoutube.com/KzB6Gb.gif](https://youtu.be/FdN4oRy5g6E)
[![Demo Doccou alpha](https://j.gifs.com/jqE9QB.gif)](https://youtu.be/FdN4oRy5g6E)

## Prerequisites
Python 3 
numpy
pytorch
tensorflow
Gym Retro
Game ROM of Sonic the Hedghog 2

## Running the code
...

### Author
Eyþór Einarsson

### Links
* [RAF117F, University of Iceland](https://ugla.hi.is/kennsluskra/index.php?sid=&tab=nam&chapter=namskeid&id=70970220186)
* [UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
* [Reinforcement Learning, book by Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/the-book-2nd.html)

