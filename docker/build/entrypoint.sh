#!/bin/bash

USER="$(whoami)"

if [ "$USER" = "datalegend" ]
then
    cd $HOME/python_venv/hypodisc

    if [ $# -gt 0 ]
    then
        pattern_file="$1"
        
        $HOME/python_venv/bin/python3 hypodisc/browse.py --suppress_browser "/mnt/shared/$pattern_file"
    fi

    /bin/bash
fi
