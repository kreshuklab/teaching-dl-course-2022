# EMBL Deep Learning course 2021/22 exercises and materials

## Schedule:

Webinar 1 (03.12.2021)

Webinar 2 (17.12.2021)

Webinar 3 (10.01.2022)


## Set-up:

We will use on-premise EMBL resources for the course. You can also use google colab to run any of the notebooks, see [webinar1](https://github.com/kreshuklab/teaching-dl-course-2022/tree/main/webinar1) for details.

## BAND

We will use BAND desktops, cloud based virtual desktops provided by EMBL to work on the exercises for Webinar 2 & 3 and during the course itself. 
These BAND desktops are open resources provided by EMBL as part of EOSCLife. For the duration of the course they will be reserved for the course participants.

### Using BAND

- Go to [band.embl.de](https://band.embl.de/#/eosc-landingpage) and click `LOGIN`.
- Use your Google account to log in. If you login for the first time, you will need to wait a few minutes to obtain a confirmation e-mail. Check out [the BAND user guide](https://docs.google.com/document/d/1TZBUsNIciGMH_g4aFj2Lu_upISxh5TV9FBMrvNDWmc8/edit) or [the BAND video tutorial](https://drive.google.com/file/d/11pbF70auGyWF-1ir2XUGM8fgiY7ETxP8/view) for more information.
- This will open the `Launch Desktops` page
    - Select a suitable configuration in `Desktop configuration`, see example in the screenshot below.
    - Click `LAUNCH` and then `yes` in the prompt
    - Click `GO TO DESKTOP` in `Running desktops`
- This will open the virtual desktop in a new tab.
    - Note that for some users starting the first virtual desktop fail and display a `CONNECTION ERROR`. If that happens for you, just terminate the desktop and start a new one; that should fix the issue.  
- You can now use this desktop to use pre-installed software or install new software in your home directory
- The home directory is persistent, i.e. everything you create in it will still be there when you launch a new desktop
- The environment for the webinar exercises is already preinstalled via jupyterLab, you can access it via `Applications->Programming->DL Course`
- This will open jupyter lab. You can test it e.g. with the notebooks at https://github.com/constantinpape/dl-teaching-resources

![band2](https://user-images.githubusercontent.com/4263537/134918656-884239a9-2951-4adb-8fad-8c4c883a65ce.png)

### More information

- Users can install software or download data into their home data, which persists in between sessions.
- For more and faster storage `/tmp` can be used. It is temporary, i.e. will be lost after the session. Hence make sure to copy data you want to keep to your home directory.
- The required software and environments can be pre-installed (see example for `DL Course` environment above) so course participants don't need to set up everything on their own.
- Traning data or other large data-sets can be shared via the EMBL S3 data storage, which can be accessed via `Applications->Data Access->Mount EMBL S3 Bucket`

## Setting up an environment on your own machine

We also provide python environments that are compatible with the course material:
- `environments/pytorch` contains the pytorch environment files (one for setting up an environment with GPU support, the other for setting up a CPU-only environment)
- `environments/keras` contains the keras environment file

You can install one of these environments with conda:
```
conda env create -f </path/to/<ENV-FILE-NAME>.yaml>
```
and then activate it via
```
conda activate <ENV-NAME>
```
If you are not familiar with conda, please refer to the [conda tutorials](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).
