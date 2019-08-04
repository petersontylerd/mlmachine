import setuptools


with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

DESCRIPTION = "Accelerate the end-to-end machine learning pipeline"
DISTNAME = "mlmachine"
LICENSE = "MIT"
# DOWNLOAD_URL = 'https://pypi.org/project/'
MAINTAINER = "Tyler Peterson"
MAINTAINER_EMAIL = "petersontylerd@gmail.com"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/Petersontylerd/mlmachine/issues",
    "Source Code": "https://github.com/Petersontylerd/mlmachine",
}
URL = "https://github.com/Petersontylerd/mlmachine"
VERSION = "0.0.14"


def setup_package():
    metadata = dict(
        name=DISTNAME,
        packages=[
            "mlmachine",
            "mlmachine.explore",
            "mlmachine.features",
            "mlmachine.model",
            "mlmachine.model.evaluate",
            "mlmachine.model.tune",
        ],
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        keywords=["Machine learning", "Data science"],
        license=LICENSE,
        url=URL,
        # download_url = DOWNLOAD_URL,
        project_urls=PROJECT_URLS,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Visualization",
            "Topic :: Software Development",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.5",
        install_requires=[i.strip() for i in open("requirements.txt").readlines()],
        # dependency_links=["https://github.com/petersontylerd/prettierplot"],
    )

    setuptools.setup(**metadata)


if __name__ == "__main__":
    setup_package()
