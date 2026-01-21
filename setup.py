from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [requirement.replace('\n', '') for requirement in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    
    return requirements

setup(
    name="doc_query_rag",
    version="0.0.1",
    author="Neehanth Reddy",
    author_email="neehanthreddy8@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
