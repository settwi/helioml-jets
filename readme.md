# HelioML | Jet Hunter

Code for manipulating Solar Jet Hunter data sets

Readme serves as an index of the code.
Each subdirectory can have its own description internally if desired

## Pre-exported data
Some cutouts are already exported on Google Drive.

here's a file index:

- [`jh.tar`](https://drive.google.com/file/d/1gqBWzjtAOG0eN_3ytKj7ogWQhaBasHcR): a tarball of all JetHunter images exported by the download script (described below)
- [`jet_cutouts.asdf`](https://drive.google.com/file/d/1iODDxvGyQPLlnr2iZ-4W9DrfGZ1m-miT): an [ASDF](https://github.com/asdf-format/asdf) file containing cutout pixel regions from the FITS files in `jh.tar`

## Requirements
- Use a Python virtual environment via e.g. [uv](https://docs.astral.sh/uv/)
- Install at least Python 3.13
- Use pip to install dependencies:
```bash
source .venv/bin/activate
# the `pip install` is one long line
uv pip install astropy numpy scipy matplotlib asdf asdf-astropy sunpy[all] shapely jupyter ipynb ipywidgets pyqt6 pyside2 regions
```

**The solar jet hunter code is not in a package; you need to clone it manually**
```bash
# Say you are in the data-acquistion folder
git clone https://github.com/somusset/SolarJetHunter_catalogue.git
ln -s SolarJetHunter_catalogue/utils .
```

## Data acquisition and preparation
The `data-acquisition` subdirectory contains a few scripts.

### `region-example.ipynb`
Example of loading in the pre-prepared ASDF file from Google Drive and plotting the bounding boxes on top of
the pixel data.

There aren't any images included on the GitHub repo to keep file sizes small.

### `convert_regions.py`
Converts the Solar Jet Hunter catalog into more standard [astropy regions](https://astropy-regions.readthedocs.io/en/stable/).

The jet hunter catalog may be acquired from the [UMN conservancy website](https://conservancy.umn.edu/items/8df6939d-362c-4e65-bea8-eb5d21d330cb):
```bash
curl -L https://conservancy.umn.edu/bitstreams/539e27e6-20fb-47a7-b9c7-2b19af910120/download > jet_clusters.json
```

The output is an ASDF file which encodes the region data.

### `download_all_jethunter_images.py`
Given the Jet Hunter catalog, download all of the jets
at the identified times within the catalog.
The jets are "clustered" into jet clusters.
Each individual jet in the SJH catalog is its own entity.
They are observed at specific times.
We use the timestamps on the images to tell [JSOC](https://docs.sunpy.org/en/stable/tutorial/acquiring_data/jsoc.html) which data to download.
Currently only the $304 \AA$ channel is downloaded.

A tarball of the files is also available on [Google Drive](https://drive.google.com/file/d/1gqBWzjtAOG0eN_3ytKj7ogWQhaBasHcR/view?usp=sharing).

### `generate_cutouts.py`
Combines the output `.asdf` file from `convert_regions.py` and cuts out
submaps from the jet hunter image catalog.
Some randomness is introduced in the submaps but it can be made reproducible
by setting the random seed.
