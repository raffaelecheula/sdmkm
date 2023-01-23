import setuptools

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

with open('README.md', 'r') as f:
    readme = f.read()

setuptools.setup(
    name="sdmkm",
    version="0.1.0",
    url="https://github.com/raffaelecheula/sdmkm.git",

    author="Raffaele Cheula",
    author_email="cheula.raffaele@gmail.com",

    description="Tools for structure-dependent microkinetic modeling.",
    long_description=readme,
    license='GPL-3.0',

    packages=[
        'sdmkm',
        'sdmkm.thermochemistry'
    ],
    package_dir={
        'sdmkm': 'sdmkm'
    },
    install_requires=requirements,
    python_requires='>=3.5, <4',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
    ],
)
