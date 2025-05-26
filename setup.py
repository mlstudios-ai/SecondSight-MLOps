from setuptools import setup, find_packages

   
"""
Run setup tool to install minimum requirements 
for the core project package
""" 
setup(
    name='enigmaai',
    description='EnigmaAI core library',
    author='Anna Huang',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pyyaml'
    ],
    python_requires='>=3.9',
)
