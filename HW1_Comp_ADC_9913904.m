%% In the name of God
% HW_Comp1-ACT
% IUT
% Alireza Qazavi
% 9913904
%% ATTENTION:
    %for ploting phase trjtories uncomment 343-346 and comment 347-350
    %for plotting phase cylinders comment 343-346 and uncomment 347-350
%%
clc;clear all;close all;
%% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%XXX pi/4-QPSK Modulation, OQPSK and MSK without consideration of noise XXX
%% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
data=[0 0 1 0 0 0 1 1 1 0 0 0 1 1 1 1]; % information sequence

figure(1)
stairs([data,data(end)], 'linewidth',3), grid on;
title('Information before Transmiting');
ylim([0 1.5])
grid on

I=2*data-1; % Data Represented at NZR form for pi/4-QPSK modulation
% even index values
I_even = I(2:2:end) ;
% odd index values
I_odd = I(1:2:end) ;

figure(2)
subplot(3,1,1);
stairs([I,I(end)], 'linewidth',3), grid on;
title('Data Represented at NZR form');
ylim([-1.5 1.5])
grid on

subplot(3,1,2);
stairs([I_even,I_even(end)], 'linewidth',3), grid on;
title('even index values');
ylim([-1.5 1.5])
grid on

subplot(3,1,3);
stairs([I_odd,I_odd(end)], 'linewidth',3), grid on;
title('odd index values');
ylim([-1.5 1.5])
grid on

Tb=1;Ts=2*Tb;
fc=2/Tb;
timeOfPls = linspace(0,Ts,1000); % Time vector for one bit information
%% XXXXXXXXXXXXXXXXXXXXXXX pi/4-QPSK modulation  XXXXXXXXXXXXXXXXXXXXXXXXXXX
S=[];
S_in=[];
S_qd=[];
%number of in-phase or qd-phase components
N = numel(data)/2;
for i=1:N
    % g(t) is a pulse with 1 amplitude for Ts = 2 Tb duration
    y1=I_even(i)*cos(2*pi*fc*timeOfPls); % inphase component
    y2=I_odd(i)*sin(2*pi*fc*timeOfPls) ;% Quadrature component
    S_in=[S_in y1]; % inphase signal vector
    S_qd=[S_qd y2]; %quadrature signal vector
    S=[S y1+y2]; % modulated signal vector
end
Tx_sig=S; % transmitting signal after modulation
t=linspace(0,Ts*N,1000*N);

figure(3)
subplot(3,1,1);
plot(t,S_in,'linewidth',1), grid on;
title(' wave form for inphase component in pi/4-QPSK modulation ');
xlabel('time(sec)');
ylabel(' amplitude(volt)');

subplot(3,1,2);
plot(t,S_qd,'linewidth',1), grid on;
title(' wave form for Quadrature component in pi/4-QPSK modulation ');
xlabel('time(sec)');
ylabel(' amplitude(volt)');

subplot(3,1,3);
plot(t,Tx_sig,'r','linewidth',1), grid on;
title(' pi/4-QPSK modulated signal (sum of inphase and Quadrature phase signals)');
xlabel('time(sec)');
ylabel(' amplitude(volt)');

%envelope of modulated signal
A = norms( [S_in;S_qd], [], 1 );hold on
plot(t,A,'g','linewidth',1),
plot(t,-A,'g','linewidth',1)
hold off

figure(4)
A = norms( [S_in;S_qd], [], 1 );grid on
plot(t,A,'g','linewidth',2),
title(' Envelope of pi/4-QPSK modulated signal ');
xlabel('time(sec)');
ylabel(' magnitude(volt)');
grid
hold off
%% XXXXXXXXXXXXXXXXXXXXXXX OQPSK modulation  XXXXXXXXXXXXXXXXXXXXXXXXXXX
%number of in-phase or qd-phase components
% N = numel(data)/2; N is 8
% sampels = 1000;
t=linspace(0,Ts*N+Ts/2,1000*(N+1/2));
% 1000*(N+1/2) is shown from fig 3.3-16 in prokis's book
% N/2 * 1000 for shifting S_qd is appeared!
S_in=zeros(1,numel(t));
S_qd=zeros(1,numel(t));
for i=1:N
    y1=I_even(i)*cos(2*pi*fc*timeOfPls); % inphase component
    y2=I_odd(i)*sin(2*pi*fc*timeOfPls) ;% Quadrature component
    % number of sampel times for each symbol time is 1000 (Ts=2 and Tb=1)
    % therefore num of sample times for each bit time is 1000/2=500
    % shifing is just implemented in indexing S_in and S_qd
    S_in((i-1)*1000+1:i*1000)=y1; % inphase signal vector
    S_qd((i-1)*1000+500+1:i*1000+500)=y2; %quadrature signal vector
end
S=S_in+S_qd; % modulated signal vector
Tx_sig=S; % transmitting signal after modulation

figure(5)
subplot(3,1,1);
plot(t,S_in,'linewidth',1), grid on;
title(' wave form for inphase component in Offset-QPSK modulation ');
xlabel('time(sec)');
ylabel(' amplitude(volt)');

subplot(3,1,2);
plot(t,S_qd,'linewidth',1), grid on;
title(' wave form for Quadrature component in Offset-QPSK modulation ');
xlabel('time(sec)');
ylabel(' amplitude(volt)');

subplot(3,1,3);
plot(t,Tx_sig,'r','linewidth',1), grid on;
title(' Offset-QPSK modulated signal (sum of inphase and Quadrature phase signals)');
xlabel('time(sec)');
ylabel(' amplitude(volt)');

A = norms( [S_in;S_qd], [], 1 );
hold on
plot(t,A,'g','linewidth',1)
plot(t,-A,'g','linewidth',1)
hold off

figure(6)
A = norms( [S_in;S_qd], [], 1 );
plot(t,A,'g','linewidth',2);grid on
title(' Envelope of Offset-QPSK modulated signal ');
xlabel('time(sec)');
ylabel(' magnitude(volt)');
%% XXXXXXXXXXXXXXXXXXXXXXX MSK modulation  XXXXXXXXXXXXXXXXXXXXXXXXXXX
%number of in-phase or qd-phase components
% N = numel(data)/2; N is 8
% sampels = 1000;
t=linspace(0,Ts*N+Ts/2,1000*(N+1/2));
% 1000*(N+1/2) is shown from fig 3.3-16 in prokis's book
% N/2 * 1000 for shifting S_qd is appeared!
S_in=zeros(1,numel(t));
S_qd=zeros(1,numel(t));
for i=1:N
    % g(t)=sin(pi*t_symbol/(2*Tb) in MSK
    y1=I_even(i).*cos(2*pi*fc*timeOfPls).*sin(pi*timeOfPls/(2*Tb));% inphase component
    y2=I_odd(i).*sin(2*pi*fc*timeOfPls).*sin(pi*timeOfPls/(2*Tb));% Quadrature component
    % number of sampel times for each symbol time is 1000 (Ts=2 and Tb=1)
    % therefore num of sample times for each bit time is 1000/2=500
    % shifing is just implemented in indexing S_in and S_qd
    S_in((i-1)*1000+1:i*1000)=y1; % inphase signal vector
    S_qd((i-1)*1000+500+1:i*1000+500)=y2; %quadrature signal vector
end
S=S_in+S_qd; % modulated signal vector
Tx_sig=S; % transmitting signal after modulation

figure(7)
subplot(3,1,1);
plot(t,S_in,'linewidth',1), grid on;
title(' wave form for inphase component in MSK modulation ');
xlabel('time(sec)');
ylabel(' amplitude(volt)');

subplot(3,1,2);
plot(t,S_qd,'linewidth',1), grid on;
title(' wave form for Quadrature component in MSK modulation ');
xlabel('time(sec)');
ylabel(' amplitude(volt)');

subplot(3,1,3);
plot(t,Tx_sig,'r','linewidth',1), grid on;
title(' MSK modulated signal (sum of inphase and Quadrature phase signals)');
xlabel('time(sec)');
ylabel(' amplitude(volt)');

% A = norms( [S_in;S_qd], [], 1 );
% hold on
% plot(t,A,'g','linewidth',1)
% plot(t,-A,'g','linewidth',1)
% hold off

figure(8)
A = norms( [S_in;S_qd], [], 1 );grid
plot(t,A,'g','linewidth',2)
title(' Envelope of MSK modulated signal ');
xlabel('time(sec)');
ylabel(' magnitude(volt)');grid on;
%% XXXXXXXXXXXXXXXXXXXXXXX PSD of CPFSK modulation  XXXXXXXXXXXXXXXXXXXXXXX
figure
i = 1;
for h = [0.2 0.5 0.7 1.2 1.3 3]
    subplot(2,3,i)
    for M = [2 4 8 16]
        k = log2(M);
        T = k * Tb;
        S_v=[];
        for f = 0:0.01:3
            A_n=[];
            y1=[];
            for n = 1:M
                a_n = sinc(f*T-0.5*(2*n-1-M)*h);
                A_n = [A_n, a_n.^2];
                for m = 1:M
                    phi_h = sin(M*pi*h)/(M*sin(pi*h));
                    alfa_n_m = pi*h*(m+n-1-M);
                    B_n_m = (cos(2*pi*f*T-alfa_n_m)-phi_h*cos(alfa_n_m))/...
                        (1+phi_h.^2-2*phi_h*cos(2*pi*f*T));
                    a_m = sinc(f*T-0.5*(2*m-1-M)*h);
                    y1=[y1,B_n_m*a_n*a_m];
                end
            end
        S_v = [S_v,1/T*(mean(A_n,'all')+2*mean(y1,'all'))];
        end
        f1=0:0.01:3;
        plot(f1.*T,S_v,'DisplayName',sprintf('M = %d',M),'LineWidth',2);
        title(sprintf('PSD for CPFSK with h = %0.1f',h));
        xlabel(' Normalized frequency fT ');ylabel('Spectral Density(W/Hz)');
        legend
        hold on;grid on
    end
    i = i+1;
end
hold off
%% XXXXXXXXXXXXXXXXXXXXXXX A vision to CPM modulation  XXXXXXXXXXXXXXXXXXXX
% plot Raised cosine and it's integration for different Ls
figure
timeOfPls = linspace(0,Ts,1000); % Time vector for one bit information
for L=1:2
t=linspace(0,Ts*L,100*L);
subplot(2,2,2*L-1)
y2=1/(2*L*Ts)*(1-cos(2*pi*t/(L*Ts)));
plot(t,y2,'DisplayName',sprintf('LRC with L = %d',L),'LineWidth',2);
xlabel('time(s)');ylabel('g(t)(volt)');
legend;grid on
subplot(2,2,2*L)
% y3 = cumsum(y2)*(t(2)-t(1));
y3 = cumtrapz(y2)*(t(2)-t(1));
t = linspace(0,Ts*L+2,100*(L+2));
y3= [y3,y3(end)*ones(1,200)];
plot(t,y3,'DisplayName',sprintf('integration of LRC with L = %d',L),'LineWidth',2);
xlabel('time(sec)');ylabel('q(t)(volt)');
legend;grid on
end
%% state machine diagrams
% L = 1
s = [1 1 2 2 3 3];
t = [2 3 1 3 1 2];
weights = [+1 -1 -1 +1 +1 -1];
names = {'phi0' 'phi1' 'phi2'};
G = digraph(s,t,weights,names);
figure;plot(G,'Layout','force','EdgeLabel',G.Edges.Weight);grid on;
title('state diagram for L = 1')
%%
data=[0 0 1 0 1 0 1 1]; % information sequence
% data = fliplr(data);
% Data Represented at I form for modulation
% Grey coding
% In = 2m-1-M, m = 1,..,M
% M = 4
I = zeros(1,numel(data));
for i = 1:numel(data)
    switch data(i)
    case 0 %BCPM
        I(i) = 1;
    case 1
        I(i) = -1;
%     case 00 %QCPM
%         I(i) = -3;
%     case 01
%         I(i) = -1;
%     case 11
%         I(i) = +1;
%     case 10
%         I(i) = +3;
    otherwise
        disp('invalid data')
    end
end
Tb=1;Ts=Tb;
h = 2/3;
figure;
for L = 1:2
    timeOfPls = linspace(0,L*Ts,L*1000);%duration of one pulse
    t_symbol = linspace(0,Ts,1000);% Time vector for one bit information
    % We have Intentionally ISI for PR, for Mermory Modulation we ecounter with
    % this ISI, This can be compenserate in RX side!
    I_cumulated=0;
    PHI_vec = [];
    for n=1:numel(data)
        if L ~= 1
            if n < 3  %according to eq 4.9.4
                I_cumulated = 0;
            else
                I_cumulated = I(n-2) + I_cumulated;
            end
            if n == 1
        PHI = pi*h/(L*Ts)*(timeOfPls-L*Ts/(2*pi)*sin(2*pi*timeOfPls/(L*Ts)))*I(1); % eq (10)
            else
        PHI = pi * h * I_cumulated + ... % eq (11)of my report
        pi*h/(L*Ts)*(timeOfPls-L*Ts/(2*pi)*sin(2*pi*timeOfPls/(L*Ts)))*I(n-1)...
      + pi*h/(L*Ts)*(timeOfPls-L*Ts/(2*pi)*sin(2*pi*timeOfPls/(L*Ts)))*I(n); % eq (10)
            end
            if n == 1
        PHI_vec(:,(n-1)*1000+1:(n+1)*1000) = PHI;
            else
        PHI = PHI + [PHI_vec(:,(n-1)*1000+1:n*1000),zeros(1,1000)];        
        PHI_vec(:,(n-1)*1000+1:(n+1)*1000) = PHI;
            end
        elseif L == 1
            if n < 2  %,according to eq 4.9.4
                I_cumulated = 0;
            else
            I_cumulated = I(n-1)+ I_cumulated;
            end
            PHI = pi*h* I_cumulated + ... % eq (11)
            pi*h/(L*Ts)*(timeOfPls-((L*Ts)/(2*pi))*sin(2*pi*timeOfPls/(L*Ts)))*I(n); % eq (10)
            PHI_vec(:,(n-1)*numel(t_symbol)+1:n*numel(timeOfPls)) = PHI;
        end
    end
    PHI_vec=rad2deg(PHI_vec);
%     PHI_vec = mod(PHI_vec,360)-180;
%     t = 0:numel(PHI_vec)-1;t=t/1000;
    t=linspace(0,Ts*numel(data),1000*numel(data));
    %for ploting phase trjtories uncomment 343-346 and comment 347-350
    %for plotting phase cylinders comment 343-346 and uncomment 347-350
    subplot(2,1,L);
    plot(t,PHI_vec(:,1:numel(data)*1000),'DisplayName',sprintf('phase of BCPM with L = %d',L),'LineWidth',2);
    xlabel('time(s)');ylabel('${\it}\phi(t,I)(degree)$','Interpreter','Latex');
    legend;grid on
%     S_in = cos(PHI_vec(:,1:numel(data)*1000)); S_qd = sin(PHI_vec(:,1:numel(data)*1000));
%     subplot(1,2,L);
%     %I have built an function for plotting helix linke phase cylender
%     PHASE_CYLINDER(S_in,S_qd,t,L)
end