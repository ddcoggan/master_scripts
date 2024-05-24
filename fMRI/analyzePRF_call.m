% function to call analyze PRF Retinotopy

% for debugging
% addpath(genpath('/home/tonglab/david/repos/mrTools-4.8.0.0'));
% addpath(genpath('/home/tonglab/david/repos/analyzePRF'));
% funcs={'/mnt/NVMe2_1TB/p013_retinotopyTongLab/derivatives/pRF/sub-M015/ses-7T1/mean_before_prf/timeseries.nii.gz'};
% mask='/mnt/NVMe2_1TB/p013_retinotopyTongLab/derivatives/ROIs/sub-M015/ses-7T1/mask_analyzed.nii.gz';
% outdir='/mnt/NVMe2_1TB/p013_retinotopyTongLab/derivatives/pRF/sub-M015/ses-7T1/mean_before_prf';
% TR=2;
% remove_outliers=1;

function analyzePRF_call(funcs, mask, outdir, TR, remove_outliers)

    
    % unzip mask if necessary, flatten, and list voxel indices
    if strcmp(mask(end-2:end), '.gz')
        gunzip(mask);
        mask = mask(1:end-3);
    end
    [mdata, mhdr] = cbiReadNifti(mask);
    vol_dim = size(mdata);
    vxs = find(reshape(mdata, [], 1) ~= 0);

    % stimulus (
    kwr = load('kayWedgeRing.mat');
    stim_radius_deg = 7;
    ppd = 51;
	total_dur = 300;
	fps = 15;
	stim_size_pix = round(ppd * stim_radius_deg * 2);
	mask_size_pix = round(size(kwr.masks,1));
    resize_factor = stim_size_pix / mask_size_pix;
    stimulus = zeros(stim_size_pix, stim_size_pix, total_dur);
	for n = 1:total_dur
        first = (n-1)*fps + 1;
        last = first + fps - 1;
        frame_idcs = first:last;
        frames = zeros(size(kwr.masks, 1), size(kwr.masks,2), fps);
        for f = 1:fps
            i = kwr.wedgeringindices(frame_idcs(f));
            if i > 0
                frames(:, :, f) = kwr.masks(:, :, i);
            end
        end
        average_frame = mean(frames, 3);
        stimulus(:,:,n) = imresize(average_frame, resize_factor);
    end

    % load timeseries data
    data = {};
    stimuli = {};
    for f = 1:numel(funcs)
        func = funcs{f};

        % unzip if necessary and flatten spatial dimensions
        if strcmp(func(end-2:end), '.gz')
            gunzip(func);
            func = func(1:end-3);
        end
        [fdata fhdr] = cbiReadNifti(func);
        fdata_flat = single(reshape(fdata, [], size(fdata,4)));

        % handle outliers
        if remove_outliers
            outlier_cutoff = 3; % in SD's for each voxel
            mean_of_voxels = mean(fdata_flat, 2, 'omitnan');
            std_of_voxels = std(fdata_flat, [], 2);
            top_each_voxel = mean_of_voxels + outlier_cutoff * std_of_voxels;
            bottom_each_voxel = mean_of_voxels - outlier_cutoff * std_of_voxels;
            all_outlier = [];
            for i = 1:size(fdata_flat,1)
                for j = 1:size(fdata_flat,2)
                    if fdata_flat(i,j) > top_each_voxel(i)
                        fdata_flat(i,j) = top_each_voxel(i);
                        all_outlier = [all_outlier; i,j];
                    elseif fdata_flat(i,j) < bottom_each_voxel(i)
                        fdata_flat(i,j) = bottom_each_voxel(i);
                        all_outlier = [all_outlier; i,j];
                    end
                end
            end
            num_outlier = length(all_outlier);
            percent_outlier = 100 * num_outlier / numel(data);
        end
        % resample to 1Hz to match stimulus (following example1.m)
        fdata_resampled = tseriesinterp(fdata_flat,2,1,2); % for TR = 2
        data{f} = fdata_resampled;

        % repeat stimulus if runs are concatenated
        num_runs = size(fdata_resampled, 2) / 300;
        stimuli{f} = repmat(stimulus, 1, num_runs);
    end
    

    % call analyzePRF function
    if isempty(gcp)
        parpool;
    end
    opts = struct('seedmode', 2, 'vxs', vxs, 'hrf', HRF_doubleGamma', ...
        'display', 'off');
    prf = analyzePRF(stimuli, data, 1, opts);
    
    % post processing
    prf.rfsize_sigma_deg = (prf.rfsize .* sqrt(prf.expt)) / ppd;
    prf.ecc_deg = prf.ecc / ppd;
    if remove_outliers
        prf.options.outlier_cutoff = outlier_cutoff;
        prf.options.all_outlier = all_outlier;
        prf.options.num_outlier = num_outlier;
        prf.options.percent_outlier = percent_outlier;
    end
    
    % save results
    save(fullfile(outdir, 'prfs'), '-struct', 'prf');

    % save out statistical maps
    fhdr.datatype = 16; 
    fhdr.bitpix = 32; % Float32 (important to preserve precision)
    cbiWriteNifti([outdir, '/polar_angle.nii'], reshape(prf.ang, vol_dim), fhdr);
    cbiWriteNifti([outdir, '/eccentricity_deg.nii'], reshape(prf.ecc_deg, vol_dim), fhdr);
    cbiWriteNifti([outdir, '/rfsize_sigma_deg.nii'], reshape(prf.rfsize_sigma_deg, vol_dim), fhdr);
    cbiWriteNifti([outdir, '/r2.nii'], reshape(prf.R2, vol_dim), fhdr);

end


