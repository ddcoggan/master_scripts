function [HIRF] = HIRF_doubleGamma_2

% Generates HIRF gamma function
%
% [HIRF] = HIRF_gamma(tau,delta);
%
% Input params:
%     tau:      time constant (sec)
%     delta:    pure delay after stimulus onset (sec)
%
% Output params:
%     HIRF:     model HRF
%
% written by Sonia Poltoratski 9/11/14 based on Lindquist et al (2009)
% Neuroimage (who takes it from SPM)
% defaults that they present are:
% alpha1 = 6
% alpha2 = 16
% beta1=beta2=1
% c = 1/6

% HRF params
% t = 0:2:30; % 30s HRF subsampled by our 2s TRs
t = 0:2:50; % 30s HRF subsampled by our 2s TRs

amplitude = 1;
alpha1 = 6;
alpha2 = 16;
beta1 = 1;
beta2 = 1;
c = 1/3;

HIRF= amplitude.*(((t.^(alpha1-1).*beta1^(alpha1).*exp(-beta1.*t))./gamma(alpha1))-c.*((t.^(alpha2-1).*beta2.^(alpha2).*exp(-beta2.*t))./gamma(alpha2)));
%figure; plot(t,HIRF,'r');