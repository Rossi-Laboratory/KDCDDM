# setup.py
#
# Project installer for: Defect Diffusion Models
# This setup script allows installation of the package and defines CLI entrypoints
# for training (ICDDM, CDLDM, KDCDDM). It includes model training, inference,
# and deployment tools for semiconductor defect detection and augmentation.

from setuptools import setup, find_packages

setup(
    name='defect_diffusion_models',
    version='0.1.0',
    description='Cross-domain diffusion models for semiconductor defect detection, augmentation, and generation.',
    long_description=(
        'This repository provides a set of cross-domain latent/pixel space diffusion models for semiconductor '
        'defect detection, data augmentation, and synthetic defect generation. It includes three training '
        'pipelines (ICDDM, CDLDM, KDCDDM), reusable model components (UNet, VAE, GAN), as well as inference '
        'and export tools for deployment. The project is modular, extensible, and ready for research or production usage.'
    ),
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourname/defect_diffusion_models',
    packages=find_packages(
        exclude=['tests', 'checkpoints', 'logs', 'samples']
    ),
    install_requires=[
        'torch>=1.10',
        'torchvision',
        'PyYAML',
        'numpy',
        'matplotlib',
        'tqdm',
        'opencv-python'
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'train_icddm=trainers.train_icddm:train_icddm',
            'train_cdldm=trainers.train_cdldm:train_cdldm',
            'train_kdcdm=trainers.train_kdcdm:train_kdcdm',
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research'
    ],
    include_package_data=True,
    zip_safe=False,
)
