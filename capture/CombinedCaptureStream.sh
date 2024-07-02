#!/bin/bash

# Parameters
outpath="."
capturetime=10
sleeptime=$((capturetime-1))
index=0

# Continuous capture stream
while true
do
    # Start capturing new av segment
    echo "" && echo "Opening capture stream #${index}"
    nohup videosnap -w 0 -t $capturetime -p "1280x720" "${outpath}/segment-${index}.mp4" &
    ((index++))

    # Wait before starting next (overlapping) segment & check for quit signal
    read -t $sleeptime -n 1 input
    if [[ ! -z $input ]]
    then
        if [[ "${input}"  == "q" ]]
        then
            exit
        elif [[ "${input}"  == "Q" ]]
        then
            exit
        fi
    fi
done
