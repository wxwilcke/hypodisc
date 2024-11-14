 # Hypodisc Docker Image

These scripts help you create and run a Docker image with Hypodisc.

## Generate image

To generate the image, first enter the `build` directory and execute the following command:

    docker build -t hypodisc:1.0.0 .

## Run Hypodisc

Once the image has been generated you can run the container using the following command:

    docker run -it -u datalegend -v <SHARED\_DIR>:/mnt/shared/ hypodisc:1.0.0

This will start the container and will place you in the directory from where you can run Hypodisc. To share files with the container (which is necessary to run Hypodisc on them) replace `<SHARED\_DIR>` with the directory on your device which contains these files. The files are now accessible from within the container at `/mnt/shared/`. 

To now run Hypodisc on your files, use a command similar to this:

    python hypodisc/run.py --depth <FROM>:<TO> --min_support <MIN\_SUPPORT> -o /mnt/shared/ /mnt/shared/<INPUT\_FILE>

The `-o /mnt/shared/` part ensures that the output is also stored in the shared directory on your device. Any files stored elsewhere in the container will be lost on exit.

## Run pattern browser

To view the output of Hypodisc in the pattern browser, place the output file in the shared directory on your device (if it is not there already) and execute the following command:

    docker run -u datalegend -v <SHARED\_DIR>:/mnt/shared/ -p 5000:5000 hypodisc:1.0.0 <PATTERN\_FILE>

and point your browser to <http://127.0.0.1:5000/viewer>
