function [ denoised ] = denoise(noisy,sigma_hat,width,height,denoiser)
% function [ denoised ] = denoise(noisy,sigma_hat,width,height,denoiser)
%DENOISE takes a signal with additive white Guassian noisy and an estimate
%of the standard deviation of that noise and applies some denosier to
%produce a denoised version of the input signal
% Input:
%       noisy       : signal to be denoised
%       sigma_hat   : estimate of the standard deviation of the noise
%       width   : width of the noisy signal
%       height  : height of the noisy signal. height=1 for 1D signals
%       denoiser: string that determines which denosier to use. e.g.
%       denoiser='BM3D'
%Output:
%       denoised   : the denoised signal.

%To apply additional denoisers within the D-AMP framework simply add
%aditional case statements to this function and modify the calls to D-AMP

noisy=reshape(noisy,[width,height]);

switch denoiser
    case 'NLM'
        if min(height,width)==1
            %Scale signal from 0 to 1 for NLM denoiser
            max_in=max(noisy(:));
            min_in=min(noisy(:));
            range_in=max_in-min_in+eps;
            noisy=(noisy-min_in)/range_in;
            Options.filterstrength=sigma_hat/range_in*1.5;
            Options.kernelratio=5;
            Options.windowratio=10;
            output=range_in*NLMF(noisy,Options)+min_in;
        else
            if sigma_hat>200
                Options.kernelratio=5;
                Options.windowratio=17;
                Options.filterstrength=sigma_hat/255*.9;
            elseif sigma_hat>150
                Options.kernelratio=5;
                Options.windowratio=17;
                Options.filterstrength=sigma_hat/255*.8;
            elseif sigma_hat>100
                Options.kernelratio=5;
                Options.windowratio=17;
                Options.filterstrength=sigma_hat/255*.6;
            elseif sigma_hat>=75
                Options.kernelratio=5;
                Options.windowratio=17;
                Options.filterstrength=sigma_hat/255*.5;
            elseif sigma_hat>=45
                Options.kernelratio=4;
                Options.windowratio=17;
                Options.filterstrength=sigma_hat/255*.6;
            elseif sigma_hat>=30
                Options.kernelratio=3;
                Options.windowratio=17;
                Options.filterstrength=sigma_hat/255*.9;
            elseif sigma_hat>=15
                Options.kernelratio=2;
                Options.windowratio=10;
                Options.filterstrength=sigma_hat/255*1;
            else
                Options.kernelratio=1;
                Options.windowratio=10;
                Options.filterstrength=sigma_hat/255*2;
            end
            Options.nThreads=4;
            Options.enablepca=true;
            output=255*NLMF(noisy/255,Options);
        end
    case 'Gauss'
        h = fspecial('gaussian',5,sigma_hat);
        output=imfilter(noisy,h,'symmetric');
    case 'Bilateral'
        Options.kernelratio=0;
        Options.blocksize=256;
        if sigma_hat>200
            Options.windowratio=17;
            Options.filterstrength=sigma_hat/255*5;
        elseif sigma_hat>150
            Options.windowratio=17;
            Options.filterstrength=sigma_hat/255*4.5;
        elseif sigma_hat>100
            Options.windowratio=17;
            Options.filterstrength=sigma_hat/255*3.5;
        elseif sigma_hat>=75
            Options.windowratio=17;
            Options.filterstrength=sigma_hat/255*3;
        elseif sigma_hat>=45
            Options.windowratio=17;
            Options.filterstrength=sigma_hat/255*2.5;
        elseif sigma_hat>=30
            Options.windowratio=17;
            Options.filterstrength=sigma_hat/255*2.2;
        elseif sigma_hat>=15
            Options.windowratio=10;
            Options.filterstrength=sigma_hat/255*2.2;
        else
            Options.windowratio=10;
            Options.filterstrength=sigma/255*2;
        end
        Options.nThreads=4;
        Options.enablepca=false;
        output=255*NLMF(noisy/255,Options);
    case 'BLS-GSM'
        %Parameters are BLS-GSM default values
        PS = ones([width,height]);
        seed = 0;
        Nsc = ceil(log2(min(width,height)) - 4);
        Nor = 3;
        repres1 = 'uw';
        repres2 = 'daub1';
        blSize = [3 3];
        parent = 0;
        boundary = 1;
        covariance = 1;
        optim = 1;
        output = denoi_BLS_GSM(noisy, sigma_hat, PS, blSize, parent, boundary, Nsc, Nor, covariance, optim, repres1, repres2, seed); 
    case 'BM3D'
        [NA, output]=BM3D(1,noisy,sigma_hat,'np',0);
        output=255*output;
    case 'fast-BM3D'
        [NA, output]=BM3D(1,noisy,sigma_hat,'lc',0);
        output=255*output;
    case 'BM3D-SAPCA'
        output = 255*BM3DSAPCA2009(noisy/255,sigma_hat/255);
    otherwise
        error('Unrecognized Denoiser')
end
    denoised=output(:);
end

