
%% To run this program, please first add the ALL-KRAKEN folder in the root directory to the path of Matlab.
clc;clear;close;


%% Sound field construction with Kraken

set(0, 'DefaultAxesFontSize', 14);
set(groot,'defaultLineLineWidth',1.5)
kraken DeepMD
kraken TrueSSP
kraken EOF
kraken KSVD

global  units
units  = 'km';

%% read shd file and plot transmission loss field
figure;
subplot(221)
plotshd('TrueSSP.shd')
caxis([10 70])
cb = colorbar;
axis([0 2 0 106])
title('Real SSP')
% 

subplot(222)
plotshd DeepMD.shd
caxis([10 70])
cb = colorbar;
axis([0 2 0 106])
title('Deep MD estimate')

subplot(223)
plotshd EOF.shd
caxis([10 70])
cb = colorbar;
axis([0 2 0 106])
title('EOF estimate')

subplot(224)
plotshd KSVD.shd
caxis([10 70])
cb = colorbar;
axis([0 2 0 106])
cb.Label.String = 'Transmission loss (dB)';
title('K-SVD estimate')

%%
idx = 80;
[~, ~, ~, ~, ~, p_True] = read_shd('TrueSSP.shd');
[~, ~, ~, ~, ~, p_deep] = read_shd('DeepMD.shd');
[~, ~, ~, ~, ~, p_eof] = read_shd('EOF.shd');
[~, ~, ~, ~, Pos, p_ksvd] = read_shd('KSVD.shd');
% Pos.r.range = Pos.r.range(1:201);

Npoints = length(Pos.r.range);
figure;
plt_True = zeros(Npoints,1);
plt_deep = zeros(Npoints,1);
plt_eof = zeros(Npoints,1);
plt_ksvd = zeros(Npoints,1);
for i = 1:1:Npoints
    plt_True(i) =  p_True(1,1,idx,i);
    plt_deep(i) =  p_deep(1,1,idx,i);
    plt_eof(i) =  p_eof(1,1,idx,i);
    plt_ksvd(i) =  p_ksvd(1,1,idx,i);
end

disrange = 2001;
subplot(131)
semilogy(Pos.r.range, abs(plt_True)); % Draw the propagation loss for line idx
hold on 
semilogy(Pos.r.range, abs(plt_deep),'--'); % Draw the propagation loss for line idx
set(gca, 'YTick', [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,1]); % Set the scale of the vertical axis
set(gca, 'YTickLabel', {'60', '50', '40', '30','20','10','0' }); % Set labels for vertical coordinates
xlabel('Range (m)')
ylabel('Transmission loss (dB)')
axis([0 disrange 0.0001 1])
rmsle_deep = sqrt(sum((log(abs(plt_True)+1)-log(abs(plt_deep)+1)).^2) / Npoints);
title(['Deep MD estimate, RMSLE = ',sprintf('%0.2e',rmsle_deep)])

subplot(132)
semilogy(Pos.r.range, abs(plt_True)); % 
hold on 
semilogy(Pos.r.range, abs(plt_eof),'--'); %
set(gca, 'YTick', [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,1]); % 
set(gca, 'YTickLabel', {'60', '50', '40', '30','20','10','0' }); % 
xlabel('Range (m)')
% ylabel('Transmission loss (dB)')
axis([0 disrange 0.0001 1])
rmsle_eof = sqrt(sum((log(abs(plt_True)+1)-log(abs(plt_eof)+1)).^2) / Npoints);
title(['EOF estimate, RMSLE = ',sprintf('%0.2e',rmsle_eof)])


subplot(133)
semilogy(Pos.r.range, abs(plt_True)); % 
hold on 
semilogy(Pos.r.range, abs(plt_ksvd),'--'); % 
set(gca, 'YTick', [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,1]); % 
set(gca, 'YTickLabel', {'60', '50', '40', '30','20','10','0' }); % 
xlabel('Range (m)')
% ylabel('Transmission loss (dB)')
axis([0 disrange 0.0001 1])
rmsle_ksvd = sqrt(sum((log(abs(plt_True)+1)-log(abs(plt_ksvd)+1)).^2) / Npoints);
title(['K-SVD estimate, RMSLE = ',sprintf('%0.2e',rmsle_ksvd)])
legend('Loss calculated by real SSP','Loss calculated by estimated SSP','location','southeast')


% 
fclose('all');
delete *.mod *.prt *.shd

