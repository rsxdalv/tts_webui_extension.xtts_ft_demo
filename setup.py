import setuptools

# todo - fix metadata
setuptools.setup(
    # name="extension_audiocraft_plus",
    name="extension_xtts_ft_demo",
    packages=setuptools.find_namespace_packages(),
    # version="2.0.3",
    version="0.0.1",
    # author="lj1995",
    author="rsxdalv",
    license="MPL-2.0",
    # description="An easy-to-use Voice Conversion framework based on VITS",
    description="XTTS fine-tuning demo",
    # url="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI",
    url="https://github.com/rsxdalv/extension_xtts_ft_demo",
    project_urls={},
    scripts=[],
    # include_package_data=True,
    # include JSON files
    # package_data={
    #     "": ["*.json"],
    # },
    install_requires=[
        # "audiocraft>=1.2",
        # "pytaglib",
        "coqui-tts==0.26.0",
        "faster_whisper",
        
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
