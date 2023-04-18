function correctBiasT1(inFile)

    addpath /home/tonglab/david/repos/spm12
    matlabbatch{1}.spm.spatial.preproc.channel.vols = {[inFile ',1']};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 18; % https://surfer.nmr.mgh.harvard.edu/fswiki/HighFieldRecon
    % matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 30; % https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg61909.html
    matlabbatch{1}.spm.spatial.preproc.channel.write = [1 1];
    matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 0;
    matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch{1}.spm.spatial.preproc.warp.samp = 2;
    matlabbatch{1}.spm.spatial.preproc.warp.write = [0 0];
    
    spm('defaults','fmri')
    spm_jobman('initcfg');
    spm_jobman('run', matlabbatch);
end
