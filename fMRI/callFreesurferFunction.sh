#!/bin/bash

while getopts s: option
do
case "${option}"
in
s) STRING=${OPTARG};;
esac
done

FSLDIR=/usr/local/fsl
. ${FSLDIR}/etc/fslconf/fsl.sh
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH
export FREESURFER_HOME=$HOME/david/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=$HOME/david/freesurfer/subjects

$STRING
