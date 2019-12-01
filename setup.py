import setuptools


with open("readme.md") as f:
    long_description = f.read()

description = "accelerate the end_to_end machine learning pipeline"
distname = "mlmachine"
license = "mit"
# download_url = 'https://pypi.org/project/'
maintainer = "tyler peterson"
maintainer_email = "petersontylerd@gmail.com"
project_urls = {
    "bug tracker": "https://github.com/petersontylerd/mlmachine/issues",
    "source code": "https://github.com/petersontylerd/mlmachine",
}
url = "https://github.com/petersontylerd/mlmachine"
version = "0.0.14"


def setup_package():
    metadata = dict(
        name=distname,
        packages=[
            "mlmachine",
            "mlmachine.datasets",
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
        classifiers=[
            "development status :: 2 - pre_alpha",
            "intended audience :: developers",
            "intended audience :: science/research",
            "topic :: scientific/engineering",
            "topic :: scientific/engineering :: artificial intelligence",
            "topic :: scientific/engineering :: information analysis",
            "topic :: scientific/engineering :: visualization",
            "topic :: software development",
            "topic :: software development :: libraries :: python modules",
            "license :: osi approved :: mit license",
            "programming language :: python :: 3",
            "operating system :: os independent",
        ],
        python_requires=">=3.5",
        install_requires=[i.strip() for i in open("requirements.txt").readlines()],
        # dependency_links=["https://github.com/petersontylerd/prettierplot"],
    )

    setuptools.setup(**metadata)


if __name__ == "__main__":
    setup_package()
