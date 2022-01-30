# Como ejecutar los ejemplos #

`git clone --recurse-submodules xxx`

`cd ps-fenics-torch`

`docker build -t ps-fenics-torch .`

`docker run -it --name="ps-fenics-torch" -v ${PWD}:/ps-fenis-torch --rm ps-fenics-torch`

Luego, por ejemplo, para ejecutar 2d_example.py:

`cd 2d_example`

`python3 2d_example.py`
