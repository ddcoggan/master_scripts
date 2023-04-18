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
$SUBJECT/mri/wm.mgz \
$SUBJECT/mri/brainmask.mgz \
$SUBJECT/mri/aseg.presurf.mgz:colormap=lut:opacity=0.2 \
-f $SUBJECT/surf/lh.white:edgecolor=blue \
$SUBJECT/surf/lh.pial:edgecolor=red \
$SUBJECT/surf/rh.white:edgecolor=blue \
$SUBJECT/surf/rh.pial:edgecolor=red
