import setuptools


with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

DESCRIPTION = 'Accelerate the end-to-end machine learning pipeline'
DISTNAME = 'mlmachine'
LICENSE = 'MIT'
# DOWNLOAD_URL = 'https://pypi.org/project/'
MAINTAINER = 'Tyler Peterson'
MAINTAINER_EMAIL = 'petersontylerd@gmail.com'
PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/Petersontylerd/mlmachine/issues'
    ,'Source Code': 'https://github.com/Petersontylerd/mlmachine'
}
URL = 'https://github.com/Petersontylerd/mlmachine'
VERSION = '0.0.1'

def setup_package():
    metadata = dict(name = DISTNAME
                    ,maintainer = MAINTAINER
                    ,maintainer_email = MAINTAINER_EMAIL
                    ,description = DESCRIPTION
                    ,keywords = ['Machine learning','Data science']
                    ,license = LICENSE
                    ,url = URL
                    ,download_url = DOWNLOAD_URL
                    ,packages = setuptools.find_packages()
                    ,project_urls = PROJECT_URLS
                    ,version = VERSION
                    ,long_description = LONG_DESCRIPTION
                    ,classifiers = ['Development Status :: 3 - Alpha'
                                    ,'Intended Audience :: Data Scientists'
                                    ,'Intended Audience :: Developers'
                                    ,'Intended Audience :: Science/Research'
                                    ,'Topic :: Machine Learning :: Data Science'
                                    ,'License :: OSI Approved :: MIT License'
                                    ,'Programming Language :: Python :: 3'
                                    ,'Topic :: Software Development'
                                    ,'Topic :: Scientific/Engineering'
                                    ,'Operating System :: OS Independent'
                                 ],
                    python_requires = ">=3.5",
                    ],
                )

    setuptools.setup(**metadata)

if __name__ == "__main__":
    setup_package()