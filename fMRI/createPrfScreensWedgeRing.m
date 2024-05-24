function screens = createPrfScreensWedgeRing(downSampleFactor, prfRadius, ppd)
% Create screens for the PRF wedge and ring stimulus outside of 7T scanner.
%
% History :
% 2018.08.16 - Created.
% 
% Copyright (C) Hojin Jang
% hoin.jang@vanderbilt.edu

load('kayWedgeRing.mat');

%%%% kay stimuli parameters, order is CCWx2 expandx2 CWx2 contrastx2
kay.wedgeLength = 32; % in seconds
kay.ringLength = 28; % in seconds
kay.postRingOff = 4; % after each ring
kay.initialFixation = 22;
kay.finalFixation = 22;
kay.totalSeconds = 300;
kay.frameRate = 1/15;

%%%%% set up mask generation
stimSize = round(ppd * prfRadius);
maskSize = round(size(masks,1));

%%%% use the stimulus mask half way through each TR for the stim fitting
midTRscreens = 1/kay.frameRate:(2*1/kay.frameRate):(kay.totalSeconds*1/kay.frameRate);

for n = 1:length(midTRscreens)
    screenNum = midTRscreens(n);
    if wedgeringindices(screenNum)>0
    WRmask = masks(:,:,wedgeringindices(screenNum));
    screen = imresize(WRmask,stimSize/maskSize)/max(WRmask(:));
    else screen = zeros(stimSize,stimSize);end
    screen = downsample(screen,downSampleFactor);
    screen = screen';
    screen = downsample(screen,downSampleFactor);
    screen = screen';
    
    screens(:,:,n) = screen;
    %F(n) = im2frame((screen+1)*10,colormap('winter'));
end

%%%% play the stimulus to check...
% implay(F)

%%%% save
% eval(['save multiBarScreens_scale' num2str(downSampleFactor) '.mat screens']);
