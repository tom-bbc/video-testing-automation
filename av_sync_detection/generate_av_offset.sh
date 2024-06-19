#!/bin/bash
infile=$1
offset=$2
outfile=$3

# Check if offset is negative (audio behind of video) or positive (audio ahead of video)
if [ 1 -eq "$(echo "$offset < 0.0" | bc)" ]
then
    # Skips forward `offset` seconds in video (meaning audio is `offset` seconds behind of video)
    offset=${offset#-}
    ffmpeg -y -i $infile -vf trim=start=$offset,setpts=PTS-STARTPTS tmp-vid-offset.mp4
else
    # Skips forward `offset` seconds in audio (meaning audio is `offset` seconds ahead of video)
    ffmpeg -y -i $infile -itsoffset $offset -i $infile -map 1:v -map 0:a -c copy tmp-vid-offset.mp4
fi

# Remove the `offset` seconds of excess content at end of clip introduced by now mismatched audio and video lengths
inputdur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $infile)
newdur=$(echo "$inputdur - $offset" | bc)
ffmpeg -y -ss 00:00:00 -accurate_seek -i tmp-vid-offset.mp4 -t $newdur -c:v libx264 -c:a flac $outfile

rm tmp-vid-offset.mp4
