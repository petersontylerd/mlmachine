import setuptools

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