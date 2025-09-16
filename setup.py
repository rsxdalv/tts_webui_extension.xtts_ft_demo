import setuptools

setuptools.setup(
    name="tts_webui_extension.xtts_ft_demo",
    packages=setuptools.find_namespace_packages(),
    version="0.0.1",
    author="rsxdalv",
    license="MPL-2.0",
    description="XTTS fine-tuning demo",
    url="https://github.com/rsxdalv/extension_xtts_ft_demo",
    project_urls={},
    scripts=[],
    install_requires=[
        "coqui-tts==0.26.0",
        "faster_whisper",
        
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
