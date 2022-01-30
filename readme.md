# Como ejecutar los ejemplos #

`git clone --recurse-submodules xxx`

`cd ps-fenics-torch`

`docker build -t gmg-top-opt .`

`docker run -it --name="gmg-top-opt" gmg-top-opt`

Luego, a modo de ejemplo, para ejecutar 2d_example.py:

`cd 2d_example`

`python3 2d_example.py`
