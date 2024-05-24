import os,glob
# all projects into single folder
for hard_file in glob.glob('/mnt/HDD1/projects_p001-p021/*'):
	link_file = f'/mnt/SSD2/projects'
	os.system(f'ln -s {hard_file} {link_file}')
for hard_file in glob.glob('/mnt/HDD2/projects/*'):
	link_file = f'/mnt/SSD2/projects'
	os.system(f'ln -s {hard_file} {link_file}')
# all data subfolders into single home folder
for hard_file in glob.glob('/mnt/SSD2/*'):
	link_file = f'/home/david/david'
	os.system(f'ln -s {hard_file} {link_file}')	
