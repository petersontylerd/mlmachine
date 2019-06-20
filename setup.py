import setuptools

DISTNAME = 'mlmachine'
DESCRIPTION = 'Accelerate the end-to-end machine learning pipeline'

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Tyler Peterseon'
MAINTAINER_EMAIL = 'petersontylerd@gmail.com'
# URL = 'http://scikit-learn.org'
# DOWNLOAD_URL = 'https://pypi.org/project/scikit-learn/#files'
LICENSE = 'MIT'
PROJECT_URLS = {
    # 'Bug Tracker': 'https://github.com/scikit-learn/scikit-learn/issues',
    # 'Documentation': 'https://scikit-learn.org/stable/documentation.html',
    'Source Code': 'https://github.com/Petersontylerd/mlmachine'
}

def setup_package():
    metadata = dict(name = DISTNAME
                    ,maintainer = MAINTAINER
                    ,maintainer_email = MAINTAINER_EMAIL
                    ,description = DESCRIPTION
                    ,license = LICENSE
                    ,url = URL
                    ,download_url = DOWNLOAD_URL
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
                                    ,'Operating System :: Microsoft :: Windows'
                                    ,'Operating System :: POSIX'
                                    ,'Operating System :: Unix'
                                    ,'Operating System :: MacOS'
                                 ],
                    python_requires = ">=3.5",
                    ],
                )

    setuptools.setup(**metadata)

setuptools.setup(
  name = 'mlmachine',
  packages = setuptools.find_packages(),
  version = '0.0.1',
  license = 'MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'TYPE YOUR DESCRIPTION HERE',   # Give a short description about your library
  author = 'Tyler Peterson',
  author_email = 'petersontylerd@gmail.com',
  url = 'https://github.com/user/reponame',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['Machine learning','Data science'],
  install_requires = [            # I get to this in a second
          'validators',
          'beautifulsoup4',
      ],
  classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Data Scientists',
    'Topic :: Machine Learning :: Data Science',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)



if __name__ == "__main__":
    setup_package()