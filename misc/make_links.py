import os,glob

for hardFile in glob.glob('/mnt/NVMe1_1TB/projects/*'):
	linkFile = f'/home/tonglab/david/projects'
	os.system(f'ln -s {hardFile} {linkFile}')
for hardFile in glob.glob('/mnt/HDD1_12TB/projects/*'):
	linkFile = f'/home/tonglab/david/projects'
	os.system(f'ln -s {hardFile} {linkFile}')
for hardFile in glob.glob('/mnt/HDD2_16TB/projects/*'):
	linkFile = f'/home/tonglab/david/projects'
	os.system(f'ln -s {hardFile} {linkFile}')
for hardFile in glob.glob('/mnt/HDD1_12TB/*'):
	if not hardFile.endswith('projects'):
		linkFile = f'/home/tonglab/david'
		os.system(f'ln -s {hardFile} {linkFile}')
	
