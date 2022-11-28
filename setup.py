import distutils.cmd
from setuptools import setup, find_packages

from utils import run_experiments


class RunExperiments(distutils.cmd.Command):
    description = 'execute paper experiments'
    user_options = [('dataset=', 'd', 'the dataset to be used ([s]/b)'),
                    ('population=', 'p', 'number of runs for the experiment ([30])'),
                    ('seed=', 's', 'seed for reproducibility ([0])')]

    def initialize_options(self) -> None:
        self.dataset = 's'
        self.population = 30
        self.seed = 0

    def finalize_options(self) -> None:
        self.population = int(self.population)
        self.seed = int(self.seed)

    def run(self) -> None:
        run_experiments(self.dataset, self.population, self.seed)


setup(
    name='kins',  # Required
    description='KINS knowledge injection algorithm test',
    license='Apache 2.0 License',
    url='https://github.com/MatteoMagnini/kins-experiments',
    author='Matteo Magnini',
    author_email='matteo.magnini@unibo.it',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='symbolic knowledge injection, ski, symbolic ai',  # Optional
    # package_dir={'': 'src'},  # Optional
    packages=find_packages(),  # Required
    include_package_data=True,
    python_requires='>=3.9.0, <3.10',
    install_requires=[
        'psyki>=0.1.10',
        'tensorflow>=2.7.0',
        'numpy>=1.22.3',
        'scikit-learn>=1.0.2',
        'pandas>=1.4.2',
    ],  # Optional
    zip_safe=False,
    cmdclass={
        'run_experiments': RunExperiments,
    },
)
