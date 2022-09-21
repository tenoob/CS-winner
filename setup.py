from setuptools import setup,find_packages
from typing import List


#declaring variables for setup function
PROJECT_NAME='housing_prediction'
VERSION='0.0.3'
AUTHOR='n00b'
DESCRIPTION = 'First full project'
PACKAGES = find_packages()  #will return the folder name which has __init__.py 
REQUIREMENT_FILE_NAME = 'requirement.txt'

def get_requirements_list()->list[str]:
    """
    Description : This function is going to return list
    of requirement mention in requirement.txt

    return This function is going to return a list which
    contain name of libraries mentioned in reqirement.txt file
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        return requirement_file.readlines().remove("-e .")


setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages=PACKAGES,
    install_requires = get_requirements_list()
)
