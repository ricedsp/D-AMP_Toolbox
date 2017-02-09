function X=SNR(s,n);

% Calcula la relacion senial a ruido en dB
%  	X=SNR(s,n);

%Calcula la energia de la se¤al (es) y del ruido (en)

es=sum(sum(abs(s).^2));
en=sum(sum(abs(n).^2));

%La relaci¢n se¤al/ruido es:
X=10*log10(es/en);

