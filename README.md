# Revisiting VCCFinder


In this repository, we provide the artefacts related to our paper *Revisiting the VCCFinder approach for the identification of vulnerability-contributing commits* to appear in the *Empirical Software Engineering* (EMSE) journal.

```bash
# clone this repository
git clone https://github.com/Trustworthy-Software/RevisitingVCCFinder.git
cd RevisitingVCCFinder
# The database dump was splited to fit in this repo
# It needs to be merged back
cat vccfinder-database.dump.xz_part* >  vccfinder-database.dump.xz
```

# Docker images
To ease replication, we provide two `Dockerfile`. 

The scripts can run *without* docker, though that requires you to set up the database and import the relevant data.

Both docker images are confirmed to run.
(In case of future issues with versions of Python libraries, we provide a file `requirements.txt_freeze` that can be used as a replacement for the current `requirements.txt` file.)
## 1 `Dockerfile`
Use that file to redo all our experiments. 

*Building* this docker image will only set up the database. 
If you want the experiments to be done at build time, uncomment the last two lines of `Dockerfile`.
Otherwise, you have to first build it, and then run it and call do_everything.sh from the docker image.

### Example:
Make sure you have already reconstructed the `vccfinder-database.dump.xz` file.
```bash
docker build -f Dockerfile -t vcc .
docker run -u root -it vcc:latest
# Now we are inside the docker container
service postgresql start
su - vcc
./do_everything.sh
```

The wole process will take ~16 hours. It runs on a machine with at least 32GB of RAM.

## 2 `Dockerfile_new_feat_only`
This file will only redo the experiments with the new features (cf. paper). 

It takes around 3h and require ~32GB of RAM.

With this Dockerfile, the experiments are done at build time (i.e., the `docker build` command will take ~3 hours). 
Feel free to comment the `RUN` commands if you'd rather run those yourself.

### Example:
Make sure you have already reconstructed the `vccfinder-database.dump.xz` file.
```bash
docker build -f Dockerfile_new_feat_only -t vcc .
```


# Paper figures
The scripts provided here perform the experiments and produce the figures found in the paper. 

The following table lists which file corresponds to which figure.

|  Ref |  -----------------   |    Location | 
| --------- | -------- |:---------|
| Figure 2 |   | replicate/4\_Results/2\_Plots/Figure2\.pdf |
| Figure 3 |   | replicate/4\_Results/2\_Plots/Figure3.pdf |
| Figure 4 |   | replicate/4\_Results/2\_Plots/Figure4.pdf |
| Figure 5 |   | new\_ft/4\_Results/2\_Plots/New\_Features\_No\_Co-Training.pdf |
| Figure 7 |   | exp\_new\_features/recall\_precision.pdf |
| Figure 8 |   | replicate/4\_Results/2\_Plots/2\_ct\_input/recall\_precision.pdf |
