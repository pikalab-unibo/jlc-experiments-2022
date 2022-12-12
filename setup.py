import distutils.cmd
from setuptools import setup, find_packages
from figures import confusion_matrix_figure, accuracy_bar_plots
from utils import run_experiments, extract_knowledge
from statistics import compute_statistics


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


class ExtractKnowledge(distutils.cmd.Command):
    description = 'extract knowledge for the wisconsin breast cancer dataset'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        extract_knowledge()


class ComputeStatistics(distutils.cmd.Command):
    description = 'compute statistics for both datasets experiments'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        compute_statistics()


class GenerateFigures(distutils.cmd.Command):
    description = 'generates figures that summarise experiments'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        confusion_matrix_figure()  # splice junction rules confusion matrix
        confusion_matrix_figure('b')  # breast cancer rules confusion matrix
        accuracy_bar_plots()
        accuracy_bar_plots('b')


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
        'psyki>=0.2.10',
        'psyke>=0.3.2.dev8'
        'tensorflow>=2.7.0',
        'numpy>=1.22.3',
        'scikit-learn>=1.0.2',
        'pandas>=1.4.2',
    ],  # Optional
    zip_safe=False,
    cmdclass={
        'run_experiments': RunExperiments,
        'extract_knowledge': ExtractKnowledge,
        'compute_statistics': ComputeStatistics,
        'generate_figures': GenerateFigures,
    },
)
