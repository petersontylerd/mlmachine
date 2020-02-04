import setuptools


with open("README.md") as f:
    long_description = f.read()

description = "Accelerate machine learning experimentation"
distname = "mlmachine"
license = "MIT"
# download_url = 'https://pypi.org/project/'
maintainer = "Tyler Peterson"
maintainer_email = "petersontylerd@gmail.com"
project_urls = {
    "bug tracker": "https://github.com/petersontylerd/mlmachine/issues",
    "source code": "https://github.com/petersontylerd/mlmachine",
}
url = "https://github.com/petersontylerd/mlmachine"
version = "0.0.33"


def setup_package():
    metadata = dict(
        name=distname,
        packages=[
            "mlmachine",
            "mlmachine.datasets",
            "mlmachine.datasets.attrition",
            "mlmachine.datasets.housing",
            "mlmachine.datasets.titanic",
            "mlmachine.explore",
            "mlmachine.features",
            "mlmachine.model",
            "mlmachine.model.evaluate",
            "mlmachine.model.explain",
            "mlmachine.model.tune",
        ],
        maintainer=maintainer,
        maintainer_email=maintainer_email,
        description=description,
        keywords=["machine learning", "data science"],
        license=license,
        url=url,
        # download_url = download_url,
        project_urls=project_urls,
        version=version,
        long_description=long_description,
        include_package_data=True,
        classifiers=[
            "Development Status :: 3 - Alpha",
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
        python_requires=">=3.6.1",
        install_requires=[i.strip() for i in open("requirements.txt").readlines()],
        # dependency_links=["https://github.com/petersontylerd/prettierplot"],
    )

    setuptools.setup(**metadata)


if __name__ == "__main__":
    setup_package()
