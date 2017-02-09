function x_hat = denoi_BLS_GSM_band(y,block,noise,prnt,covariance,optim,sig);

% It solves for the BLS global optimum solution, using a flat (pseudo)prior for log(z)
% 		  x_hat = denoi_BLS_GSM_band(y,block,noise,prnt,covariance,optim,sig);
%
%       prnt:  Include/ Not Include parent (1/0)
%       covariance: Include / Not Include covariance in the GSM model (1/0)
%       optim: BLS / MAP-Wiener(2-step) (1/0)

% JPM, Univ. de Granada, 5/02
% Last revision: JPM, 4/03


if ~exist('covariance'),
        covariance = 1;
end

if ~exist('optim'),
        optim = 1;
end

[nv,nh,nb] = size(y);

nblv = nv-block(1)+1;	% Discard the outer coefficients 
nblh = nh-block(2)+1;   % for the reference (centrral) coefficients (to avoid boundary effects)
nexp = nblv*nblh;			% number of coefficients considered
zM = zeros(nv,nh);		% hidden variable z
x_hat = zeros(nv,nh);	% coefficient estimation
N = prod(block) + prnt; % size of the neighborhood

Ly = (block(1)-1)/2;		% block(1) and block(2) must be odd!
Lx = (block(2)-1)/2;
if (Ly~=floor(Ly))|(Lx~=floor(Lx)),
   error('Spatial dimensions of neighborhood must be odd!');
end   
cent = floor((prod(block)+1)/2);	% reference coefficient in the neighborhood 
                                 % (central coef in the fine band)

Y = zeros(nexp,N);		% It will be the observed signal (rearranged in nexp neighborhoods)
W = zeros(nexp,N);		% It will be a signal with the same autocorrelation as the noise

foo = zeros(nexp,N);

% Compute covariance of noise from 'noise'
n = 0;
for ny = -Ly:Ly,	% spatial neighbors
	for nx = -Lx:Lx,
		n = n + 1;
		foo = shift(noise(:,:,1),[ny nx]);
		foo = foo(Ly+1:Ly+nblv,Lx+1:Lx+nblh);
		W(:,n) = vector(foo);
	end
end
if prnt,	% parent
	n = n + 1;
	foo = noise(:,:,2);
	foo = foo(Ly+1:Ly+nblv,Lx+1:Lx+nblh);
	W(:,n) = vector(foo);
end

C_w = innerProd(W)/nexp;
sig2 = mean(diag(C_w(1:N-prnt,1:N-prnt)));	% noise variance in the (fine) subband

clear W;
if ~covariance,
   if prnt,
        C_w = diag([sig2*ones(N-prnt,1);C_w(N,N)]);
   else
        C_w = diag(sig2*ones(N,1));
   end
end    


% Rearrange observed samples in 'nexp' neighborhoods 
n = 0;
for ny=-Ly:Ly,	% spatial neighbors
	for nx=-Lx:Lx,
		n = n + 1;
		foo = shift(y(:,:,1),[ny nx]);
		foo = foo(Ly+1:Ly+nblv,Lx+1:Lx+nblh);
		Y(:,n) = vector(foo);
	end
end
if prnt,	% parent
	n = n + 1;
	foo = y(:,:,2);
	foo = foo(Ly+1:Ly+nblv,Lx+1:Lx+nblh);
	Y(:,n) = vector(foo);
end
clear foo

% For modulating the local stdv of noise
if exist('sig') & prod(size(sig))>1,
    sig = max(sig,sqrt(1/12));   % Minimum stdv in quantified (integer) pixels
    subSampleFactor = log2(sqrt(prod(size(sig))/(nv*nh)));
    zW = blurDn(reshape(sig, size(zM)*2^subSampleFactor)/2^subSampleFactor,subSampleFactor);
    zW = zW.^2;
    zW = zW/mean2(zW); % Expectation{zW} = 1
    z_w = vector(zW(Ly+1:Ly+nblv,Lx+1:Lx+nblh));
end    

[S,dd] = eig(C_w);
S = S*real(sqrt(dd));	% S*S' = C_w
iS = pinv(S);
clear noise

C_y = innerProd(Y)/nexp;
sy2 = mean(diag(C_y(1:N-prnt,1:N-prnt))); % observed (signal + noise) variance in the subband
C_x = C_y - C_w;			% as signal and noise are assumed to be independent
[Q,L] = eig(C_x);
% correct possible negative eigenvalues, without changing the overall variance
L = diag(diag(L).*(diag(L)>0))*sum(diag(L))/(sum(diag(L).*(diag(L)>0))+(sum(diag(L).*(diag(L)>0))==0));
C_x = Q*L*Q';
   
sx2 = sy2 - sig2;			% estimated signal variance in the subband
sx2 = sx2.*(sx2>0); % + (sx2<=0); 
if ~covariance,
   if prnt,
        C_x = diag([sx2*ones(N-prnt,1);C_x(N,N)]);
   else
        C_x = diag(sx2*ones(N,1));
   end
end    
[Q,L] = eig(iS*C_x*iS');	 	% Double diagonalization of signal and noise
la = diag(L);						% eigenvalues: energy in the new represetnation.
la = real(la).*(real(la)>0);

% Linearly transform the observations, and keep the quadratic values (we do not model phase)

V = Q'*iS*Y';
clear Y;
V2 = (V.^2).';
M = S*Q;
m = M(cent,:);


% Compute p(Y|log(z))

if 1,   % non-informative prior
    lzmin = -20.5;
    lzmax = 3.5;
    step = 2;
else    % gamma prior for 1/z
    lzmin = -6;
    lzmax = 4;
    step = 0.5;
end    

lzi = lzmin:step:lzmax;
nsamp_z = length(lzi);
zi = exp(lzi);
 

laz = la*zi;
p_lz = zeros(nexp,nsamp_z);
mu_x = zeros(nexp,nsamp_z);

if ~exist('z_w'),       % Spatially invariant noise
    pg1_lz = 1./sqrt(prod(1 + laz,1));	% normalization term (depends on z, but not on Y)
    aux = exp(-0.5*V2*(1./(1+laz)));
    p_lz = aux*diag(pg1_lz);				% That gives us the conditional Gaussian density values
    										% for the observed samples and the considered samples of z
    % Compute mu_x(z) = E{x|log(z),Y}
    aux = diag(m)*(laz./(1 + laz));	% Remember: laz = la*zi
    mu_x = V.'*aux;			% Wiener estimation, for each considered sample of z
else                    % Spatially variant noise
    rep_z_w = repmat(z_w, 1, N); 
    for n_z = 1:nsamp_z,
        rep_laz = repmat(laz(:,n_z).',nexp,1);
        aux = rep_laz + rep_z_w;     % lambda*z_u + z_w
        p_lz(:,n_z) = exp(-0.5*sum(V2./aux,2))./sqrt(prod(aux,2));
        % Compute mu_x(z) = E{x|log(z),Y,z_w}
        aux = rep_laz./aux;
        mu_x(:,n_z) = (V.'.*aux)*m.';
    end
end    
                                            
                                            
[foo, ind] = max(p_lz.');	% We use ML estimation of z only for the boundaries.
clear foo
if prod(size(ind)) == 0,
	z = ones(1,size(ind,2));
else
	z = zi(ind).';				
end

clear V2 aux

% For boundary handling

uv=1+Ly;
lh=1+Lx;
dv=nblv+Ly;
rh=nblh+Lx;
ul1=ones(uv,lh);
u1=ones(uv-1,1);
l1=ones(1,lh-1);
ur1=ul1;
dl1=ul1;
dr1=ul1;
d1=u1;
r1=l1;

zM(uv:dv,lh:rh) = reshape(z,nblv,nblh);

% Propagation of the ML-estimated z to the boundaries

% a) Corners
zM(1:uv,1:lh)=zM(uv,lh)*ul1;
zM(1:uv,rh:nh)=zM(uv,rh)*ur1;
zM(dv:nv,1:lh)=zM(dv,lh)*dl1;
zM(dv:nv,rh:nh)=zM(dv,rh)*dr1;
% b) Bands
zM(1:uv-1,lh+1:rh-1)=u1*zM(uv,lh+1:rh-1);
zM(dv+1:nv,lh+1:rh-1)=d1*zM(dv,lh+1:rh-1);
zM(uv+1:dv-1,1:lh-1)=zM(uv+1:dv-1,lh)*l1;
zM(uv+1:dv-1,rh+1:nh)=zM(uv+1:dv-1,rh)*r1;

% We do scalar Wiener for the boundary coefficients
if exist('z_w'),
    x_hat = y(:,:,1).*(sx2*zM)./(sx2*zM + sig2*zW);
else        
    x_hat = y(:,:,1).*(sx2*zM)./(sx2*zM + sig2);
end


% Prior for log(z)

p_z = ones(nsamp_z,1);    % Flat log-prior (non-informative for GSM)
p_z = p_z/sum(p_z);


% Compute p(log(z)|Y) from p(Y|log(z)) and p(log(z)) (Bayes Rule)

p_lz_y = p_lz*diag(p_z);
clear p_lz
if ~optim,
   p_lz_y = (p_lz_y==max(p_lz_y')'*ones(1,size(p_lz_y,2))); 	% ML in log(z): it becomes a delta function																	% at the maximum
end    
aux = sum(p_lz_y, 2);
if any(aux==0),
    foo = aux==0;
    p_lz_y = repmat(~foo,1,nsamp_z).*p_lz_y./repmat(aux + foo,1,nsamp_z)...
        + repmat(foo,1,nsamp_z).*repmat(p_z',nexp,1); 	% Normalizing: p(log(z)|Y)
else
    p_lz_y = p_lz_y./repmat(aux,1,nsamp_z); 	% Normalizing: p(log(z)|Y)
end    
clear aux;

% Compute E{x|Y} = int_log(z){ E{x|log(z),Y} p(log(z)|Y) d(log(z)) }

aux = sum(mu_x.*p_lz_y, 2);

x_hat(1+Ly:nblv+Ly,1+Lx:nblh+Lx) = reshape(aux,nblv,nblh);

clear mu_x p_lz_y aux