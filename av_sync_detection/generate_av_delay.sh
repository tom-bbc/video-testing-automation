#!/bin/bash
infile=$1
offset=$2
outfile=$3

ffmpeg -y -i $infile -vf trim=start=$offset,setpts=PTS-STARTPTS tmp-vid-offset.mp4

dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$1")
newdur=$(echo "$dur - $offset" | bc)

ffmpeg -y -ss 00:00:00 -accurate_seek -i tmp-vid-offset.mp4 -t $newdur -c:v libx264 -c:a flac $outfile

rm tmp-vid-offset.mp4
