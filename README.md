# Latent-Space-Visualisation-PCA-t-SNE-UMAP

## What is this repo

This repository contains all the code used to generate the animations for the video ["Latent Space Visualization: PCA, t-SNE, UMAP"](https://youtu.be/o_cAOa5fMhE) on the youtube channel [Deepia](https://www.youtube.com/@Deepia-ls2fo).

All references to the script and the voiceover service have been removed, so you are left with only raw animations.

There is no guarantee that any of these scripts can actually run. The code was not meant to be re-used, so it might need some external data in some places that I may or may have not put in this repository.

You can reuse any piece of code to make the same visualizations, crediting the youtube channel [Deepia](https://www.youtube.com/@Deepia-ls2fo) would be nice but is not required.

## Environment

The environment I used to create this video is described in the ``environment.yaml`` file.
You should use conda to replicate this environment using the following command:

```bash
conda env create -f environment.yaml
```

## Generate a scene

You should move to one of the subfolders then run the regular Manim commands to generate a video such as:

```bash
manim -pqh scene_1.py --disable_caching
```

The video will then be written in the ``./media/videos/scene_1/1080p60/`` subdirectory.

I recommand the ``--disable_caching`` flag when using voiceover.
