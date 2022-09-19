# CS-winner

Improvement on existing CS:GO project v2.

to build docker img
docker buid -t <img_name>:<tagname> <location>

to run docker 
docker run -p 5000:5000 -e PORT=5000 <image id>

to check running docker containers
docker ps

to stop docker container
docker stop <container_id>
 

 to list docker img
 docker images


 in requrirement.txt "-e ." is used so that the housing package/folder can be install which has __init__.py file
  
 in setup.py we are removing "-e ." as for installing libraries we are using find_packages and both(find_packages,-e .) does the same thing so for that we are removing it

 if we are not running setup.py then just run "pip install -r requirement.txt"