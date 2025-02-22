# mooncoachAI
Moonboards are a great tool for a climber to get stronger and work on technique. A moonboard consists of a grid of standardized holds, oriented at a fixed angle, that uses lights to indicate which holds are "active" during the climb. A problem begins with hands on the starting holds illuminated green, and feet on any of the identical holds in the foot box (if permitted by the problem). A climb is complete when both hands match on the final hold, illuminated by red, using any of the highlighted blue holds.

The goal of this project is to build a neural network, trained off of climbing YouTube videos, that can solve a moonboard problem. 
So far, the data scraper is mostly finished and is able to process climbing footage and extract a sequence of moves from a climb.

The next step is to build the model to solve a given moonboard problem which I am currently working on. An example for using the datascraper
can be found in src/datascraper/main.py. 

More details and updates can be found on my website https://dchurchill.me/mooncoachAI/ as I develop the code.
