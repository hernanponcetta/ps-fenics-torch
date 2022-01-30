# FEniCS-Torch: optimización topológica #

## Como ejecutar los ejemplos ##

La forma recomendada de ejecutar los ejemplos es utilizando un contenerdor Docker, las instrucciones de instalación se pueden consultar en [Get Docker](https://docs.docker.com/get-docker/)

Un vez instalado Docker se puden ejecutar los ejemplos de la siguiente manera:

`git clone --recurse-submodules https://github.com/hernanponcetta/ps-fenics-torch.git`

`cd ps-fenics-torch`

`docker build -t ps-fenics-torch .`

`docker run -it --name=ps-fenics-torch -v ${PWD}:/home/ps-fenics-torch --rm ps-fenics-torch`

Luego, por ejemplo, para ejecutar 2d_example.py:

`cd 2d_example`

`python3 2d_example.py`

Para salir y remover el contanedor:

`exit`
