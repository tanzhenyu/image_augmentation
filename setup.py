import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="image_augmentation",
    version="0.1.0",
    author="Swarup Ghosh, Zhenyu Tan",
    author_email="snwg@live.com, tanzheny@google.com",
    description="TensorFlow implementation of AutoAugment and RandAugment.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tanzhenyu/image_augmentation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Information Technology",
        "Topic :: Utilities"
    ],
    python_requires='>=3.5',
)
