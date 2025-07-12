from setuptools import setup, find_packages

# 读取 README.md 作为长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='deepadvancer',
    version='0.1.0',
    description='A batch correction and signal deconvolution toolkit for bulk transcriptomic data using adversarial autoencoder',
    long_description=long_description,
    long_description_content_type='text/markdown',

    author='Mintian Cui',
    author_email='1308318910@qq.com',
    url='https://github.com/BioinAI/deepadvancer',

    packages=find_packages(include=['deepadvancer', 'deepadvancer.*']),
    include_package_data=True,  
    package_data={'deepadvancer': ['*.R']}, 
    install_requires=[
        'torch>=2.0,<2.4',
        'numpy>=1.26,<1.27',
        'pandas>=2.2,<2.3',
        'tqdm>=4.66,<4.67',
        'matplotlib>=3.6,<3.7',
        'scikit-learn>=1.4,<1.5',
        'scipy>=1.12,<1.13',
        'rpy2>=3.6,<3.7'
    ],
    python_requires='>=3.9',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],

    keywords='bulk RNA-seq batch-correction autoencoder deconvolution bioinformatics adversarial deep learning',
    license='MIT',
)

