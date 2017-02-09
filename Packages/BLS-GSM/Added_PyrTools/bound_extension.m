function im_ext = bound_extension(im,By,Bx,type);

% im_ext = bound_extension(im,B,type);
%
% Extend an image for avoiding boundary artifacts,
%
%   By, Bx:    widths of the added stripes.
%   type:   'mirror'        Mirror extension
%           'mirror_nr':    Mirror without repeating the last pixel
%           'circular':     fft2-like
%           'zeros'

% Javier Portilla, Universidad de Granada, Jan 2004

[Ny,Nx,Nc] = size(im);

im_ext = zeros(Ny+2*By,Nx+2*Bx,Nc);
im_ext(By+1:Ny+By,Bx+1:Nx+Bx,:) = im;

if strcmp(type,'mirror'),

    im_ext(1:By,:,:) = im_ext(2*By:-1:By+1,:,:);
    im_ext(:,1:Bx,:) = im_ext(:,2*Bx:-1:Bx+1,:);
    im_ext(Ny+1+By:Ny+2*By,:,:) = im_ext(Ny+By:-1:Ny+1,:,:);
    im_ext(:,Nx+1+Bx:Nx+2*Bx,:) = im_ext(:,Nx+Bx:-1:Nx+1,:);
    im_ext(1:By,1:Bx,:) = im_ext(2*By:-1:By+1,2*Bx:-1:Bx+1,:);
    im_ext(Ny+1+By:Ny+2*By,Nx+1+Bx:Nx+2*Bx,:) = im_ext(Ny+By:-1:Ny+1,Nx+Bx:-1:Nx+1,:);
    im_ext(1:By,Nx+1+Bx:Nx+2*Bx,:) = im_ext(2*By:-1:By+1,Nx+Bx:-1:Nx+1,:);
    im_ext(Ny+1+By:Ny+2*By,1:Bx,:) = im_ext(Ny+By:-1:Ny+1,2*Bx:-1:Bx+1,:);

elseif strcmp(type,'mirror_nr'),    
        
    im_ext(1:By,:,:) = im_ext(2*By+1:-1:By+2,:,:);
    im_ext(:,1:Bx,:) = im_ext(:,2*Bx+1:-1:Bx+2,:);
    im_ext(Ny+1+By:Ny+2*By,:,:) = im_ext(Ny+By-1:-1:Ny,:,:);
    im_ext(:,Nx+1+Bx:Nx+2*Bx,:) = im_ext(:,Nx+Bx-1:-1:Nx,:);
    im_ext(1:By,1:Bx,:) = im_ext(2*By+1:-1:By+2,2*Bx+1:-1:Bx+2,:);
    im_ext(Ny+1+By:Ny+2*By,Nx+1+Bx:Nx+2*Bx,:) = im_ext(Ny+By-1:-1:Ny,Nx+Bx-1:-1:Nx,:);
    im_ext(1:By,Nx+1+Bx:Nx+2*Bx,:) = im_ext(2*By+1:-1:By+2,Nx+Bx-1:-1:Nx,:);
    im_ext(Ny+1+By:Ny+2*By,1:Bx,:) = im_ext(Ny+By-1:-1:Ny,2*Bx+1:-1:Bx+2,:);
        
elseif strcmp(type,'circular'),        
        
    im_ext(1:By,:,:) =  im_ext(Ny+1:Ny+By,:,:);
    im_ext(:,1:Bx,:) = im_ext(:,Nx+1:Nx+Bx,:);
    im_ext(Ny+1+By:Ny+2*By,:,:) = im_ext(By+1:2*By,:,:);
    im_ext(:,Nx+1+Bx:Nx+2*Bx,:) = im_ext(:,Bx+1:2*Bx,:);
    im_ext(1:By,1:Bx,:) = im_ext(Ny+1:Ny+By,Nx+1:Nx+Bx,:);
    im_ext(Ny+1+By:Ny+2*By,Nx+1+Bx:Nx+2*Bx,:) = im_ext(By+1:2*By,Bx+1:2*Bx,:);
    im_ext(1:By,Nx+1+Bx:Nx+2*Bx,:) = im_ext(Ny+1:Ny+By,Bx+1:2*Bx,:);
    im_ext(Ny+1+By:Ny+2*By,1:Bx,:) = im_ext(By+1:2*By,Nx+1:Nx+Bx,:);
   
end    
        