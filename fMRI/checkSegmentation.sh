#!/bin/bash

# usage - in terminal type the following:
# <path/to/this/script> -s <subject ID as shown in $SUBJECTS_DIR> 

while getopts s: option
do
case "${option}"
in
s) SUBJECT=${OPTARG};;
esac
done

cd $SUBJECTS_DIR
freeview -v \
$SUBJECT/mri/T1.mgz \
$SUBJECT/mri/wm.mgz:colormap=heat:opacity=0.40:heatscale=100,250 \
$SUBJECT/mri/brainmask.mgz:visible=0 \
-f $SUBJECT/surf/lh.smoothwm:edgecolor=blue \
$SUBJECT/surf/lh.pial.T1:edgecolor=yellow \
$SUBJECT/surf/rh.smoothwm:edgecolor=blue \
$SUBJECT/surf/rh.pial.T1:edgecolor=yellow
