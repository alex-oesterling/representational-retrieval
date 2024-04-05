# representational-retrieval

To run, clone the repo

```
git clone git@github.com:alex-oesterling/representational-retrieval.git
cd representational-retrieval
pip install -e .
```
(The -e flag is important, this means editable mode so any changes you make to python inside the `representational_retrieval` folder should actually work instead of requiring you to reinstall the package each time)  

Then you can write experiments in the `experiments` folder and just type `import representational_retrieval` to access all the utility functions such as the CelebA dataloader and whatever else we add there.

Note: You will have to download CelebA, the torchvision version isnt working because of a checksum issue so I would try downloading the original Google Drive version and unzip properly. You'll have to change the path too.