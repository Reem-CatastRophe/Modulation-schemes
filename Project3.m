%% Clear Section---------------------------------
% Clear the workspace, command window, and close all figures. This section
% ensures a clean slate before running the simulations.
clear; clc; close all;

%% Bit Generation
% Number of bits to be generated
Bits_number = 100000;

% Range of SNR values in dB
SNR_dB = -4:8;  % SNR in dB

% Generate random binary data (0s and 1s)
Random_bits = randi([0, 1], 1, Bits_number);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BPSK Modulation Scheme
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BPSK bit energy
BPSK_Bit_energy = 1;

% Map binary data to BPSK symbols using bipolar (+1/-1) representation
BPSK_Mapped_bits = 2 * Random_bits - 1;  % Convert 0s to -1 and 1s to +1

%% AWGN Channel Simulation Demapping and BER Calc


% Pre-allocate array for storing BER values
BPSK_BER = zeros(size(SNR_dB));

% Pre-calculate theoretical BER for efficiency
BPSK_Theoretical_BER = 0.5 * erfc(sqrt(10.^(SNR_dB / 10)));

for BPSK_i = 1:length(SNR_dB)
    % Current SNR_dB value in dB
    current_SNR_dB = SNR_dB(BPSK_i);

    % Calculate noise power spectral density (PSD) from SNR and bit energy
    BPSK_Noise_Power_Spectral_Density = BPSK_Bit_energy / (10^(current_SNR_dB / 10));

    % Generate AWGN noise with specified PSD
    BPSK_AWGN = randn(1, Bits_number) * sqrt(BPSK_Noise_Power_Spectral_Density / 2);

    % Create received signal by adding noise to BPSK symbols
    BPSK_Received_signal = BPSK_Mapped_bits + BPSK_AWGN;

    % Demap BPSK symbols back to binary data using threshold decision
    BPSK_Demapped_signal = BPSK_Received_signal >= 0;  % +1 maps to 1, -1 maps to 0

    % Calculate bit error rate (BER) as the proportion of bit errors
    BPSK_BER(BPSK_i) = mean(BPSK_Demapped_signal ~= Random_bits);
end

%% Plotting Results

figure;
semilogy(SNR_dB, BPSK_BER, '-', 'linewidth', 2);
hold on;
semilogy(SNR_dB, BPSK_Theoretical_BER, '-', 'linewidth', 2);
xlabel('BPSK Eb/No (dB)');
ylabel('BPSK BER');
legend('Simulated BPSK BER', 'Theoretical BPSK BER');
grid on;
title('BPSK Modulation Performance');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% QPSK Modulation Scheme
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%---------------- 3)QPSK Grey Coded-----------------------%
%%%%%%%%%%%%%%%%%%%% Mapper %%%%%%%%%%%%%%%%%%%%%
% This section maps two bits from the binary data stream to a complex
% symbol for QPSK modulation using Gray coding.

QPSK_Mapped_bits = zeros(1 , (Bits_number)/2) ;  % Pre-allocate memory for efficiency

for i = 1 : (Bits_number-1)/2  % Loop through every two bits
    if isequal(Random_bits(i*2-1 : i*2), [1 1])
        QPSK_Mapped_bits(i) = (cos(0)+1i*sin(pi/2));  % Gray code: 11 maps to 0 + pi/2
    elseif isequal(Random_bits(i*2-1 : i*2), [0 1])
        QPSK_Mapped_bits(i) = (cos(pi)+1i*sin(pi/2));  % Gray code: 01 maps to pi + pi/2
    elseif isequal(Random_bits(i*2-1 : i*2), [1 0])
        QPSK_Mapped_bits(i) = (cos(0)+1i*sin(3*pi/2)); % Gray code: 10 maps to 0 + 3*pi/2
    elseif isequal(Random_bits(i*2-1 : i*2), [0 0])
        QPSK_Mapped_bits(i) = (cos(pi)+1i*sin(3*pi/2)); % Gray code: 00 maps to pi + 3*pi/2
    end
end

%%%%%%%%%%%%%%%%%%%% Channel %%%%%%%%%%%%%%%%%%%%%%%%

% Define signal properties
QPSK_Bit_energy=1;
QPSK_Noise_Power_Spectral_Density = QPSK_Bit_energy./(10.^(SNR_dB/10));

% Initialize BER arrays
QPSK_BER = zeros(1, length(SNR_dB));
QPSK_Theoretical_BER = zeros(1, length(SNR_dB));

for i = 1 : length(SNR_dB)
    % Simulate AWGN channel noise
    QPSK_AWGN = randn(1,(Bits_number)/2)*sqrt(QPSK_Noise_Power_Spectral_Density(i)/2)+ ...
        1i.*randn(1,(Bits_number)/2)*sqrt(QPSK_Noise_Power_Spectral_Density(i)/2);

    % Add noise to the transmitted signal
    QPSK_Received_signal = QPSK_Mapped_bits + QPSK_AWGN ;

    %%%%%%%%%%%%%%%%%% Demapper %%%%%%%%%%%%%%%%
    % This section recovers the original data bits from the received noisy
    % signal

    QPSK_Demapped_signal = zeros(1, Bits_number);
    for k = 1:Bits_number/2
        % Make decisions based on real and imaginary parts of received
        % signal
        if real(QPSK_Received_signal(k)) >= 0
            QPSK_Demapped_signal((k-1)*2+1) = 1;
        end
        if imag(QPSK_Received_signal(k)) >= 0
            QPSK_Demapped_signal((k-1)*2+2) = 1;
        end
    end

    %%%%%%%%%%%%%%%%%%%%% BER calculation %%%%%%%%%%%%%%%%
    % Calculate Bit Error Rate (BER)

    QPSK_Error_Grey = abs(QPSK_Demapped_signal - Random_bits);
    QPSK_BER(i) = sum(QPSK_Error_Grey)/Bits_number;

    % Calculate theoretical BER for comparison
    QPSK_Theoretical_BER(i) = (0.5)*erfc(sqrt(1/QPSK_Noise_Power_Spectral_Density(i)));
end

%%%%%%%%%%%%%% plotting BER of QPSK %%%%%%%%%%%%%%%%%
figure(3)
semilogy(SNR_dB,QPSK_BER , '-','linewidth',2 ) ;
hold on
semilogy( SNR_dB, QPSK_Theoretical_BER ,'-','linewidth',2) ;
xlabel('Eb/No');
ylabel('BER');
legend('QPSK BER' , 'Theoretical BER ') ;
grid on
title('QPSK Grey-coded Modulation');

%---------------- 3)QPSK Binary Coded-----------------------%
%%%%%%%%%%%%%%%%%%% Mapper %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This section maps two bits from the binary data stream to a complex
% symbol for QPSK modulation using a binary mapping scheme.

QPSK_Binary_Mapped_bits = zeros(1, Bits_number/2);
for i = 1:(Bits_number-1)/2  % Loop through every two bits
    if isequal(Random_bits(i*2-1 : i*2), [1 0])
        QPSK_Binary_Mapped_bits(i) = (cos(0)+1i*sin(pi/2));  % Map "10" to +j
    elseif isequal(Random_bits(i*2-1 : i*2), [0 1])
        QPSK_Binary_Mapped_bits(i) = (cos(pi)+1i*sin(pi/2));  % Map "01" to +i
    elseif isequal(Random_bits(i*2-1 : i*2), [1 1])
        QPSK_Binary_Mapped_bits(i) = (cos(0)+1i*sin(3*pi/2)); % Map "11" to -j
    elseif isequal(Random_bits(i*2-1 : i*2), [0 0])
        QPSK_Binary_Mapped_bits(i) = (cos(pi)+1i*sin(3*pi/2)); % Map "00" to -i
    end
end

%%%%%%%%%%%%%%%%%%%% channel %%%%%%%%%%%%%%%%%%%%%%%%

% Define signal properties
QPSK_Binary_Bit_energy = 1;
QPSK_Binary_Noise_Power_Spectral_Density = QPSK_Binary_Bit_energy./(10.^(SNR_dB/10));

% Initialize BER arrays
QPSK_Binary_BER = zeros(1, length(SNR_dB));
QPSK_Binary_Theoretical_BER = zeros(1, length(SNR_dB));

for i = 1:length(SNR_dB)
    % Simulate AWGN channel noise
    AWGN_QPSK_Binary = randn(1, Bits_number/2)*sqrt(QPSK_Binary_Noise_Power_Spectral_Density(i)/2) + ...
        1i.*randn(1, Bits_number/2)*sqrt(QPSK_Binary_Noise_Power_Spectral_Density(i)/2);

    % Add noise to the transmitted signal
    QPSK_Binary_Received_signal = QPSK_Binary_Mapped_bits + AWGN_QPSK_Binary;

    %%%%%%%%%%%%%%%%%% Demapper %%%%%%%%%%%%%%%%%%%%%%%%%
    % This section recovers the original data bits from the received noisy
    % signal

    QPSK_Binary_Demapped_signal = zeros(1, Bits_number);
    for k = 1:Bits_number/2
        % Make decisions based on real and imaginary parts of received
        % signal
        index = k*2 - 1;  % Simplified index calculation for bit pair
        if real(QPSK_Binary_Received_signal(k)) >= 0 && imag(QPSK_Binary_Received_signal(k)) >= 0
            QPSK_Binary_Demapped_signal(index:index+1) = [1 0];
        elseif real(QPSK_Binary_Received_signal(k)) >= 0 && imag(QPSK_Binary_Received_signal(k)) < 0
            QPSK_Binary_Demapped_signal(index:index+1) = [1 1];
        elseif real(QPSK_Binary_Received_signal(k)) < 0 && imag(QPSK_Binary_Received_signal(k)) >= 0
            QPSK_Binary_Demapped_signal(index:index+1) = [0 1];
        elseif real(QPSK_Binary_Received_signal(k)) < 0 && imag(QPSK_Binary_Received_signal(k)) < 0
            QPSK_Binary_Demapped_signal(index:index+1) = [0 0];
        end
    end

    %%%%%%%%%%%%%%%%%%%%% BER calculation %%%%%%%%%%%%%%%%
    % Calculate Bit Error Rate (BER)

    QPSK_Error_Binary = abs(QPSK_Binary_Demapped_signal - Random_bits);
    QPSK_Binary_BER(i) = sum(QPSK_Error_Binary)/Bits_number;
    % Calculate theoretical BER for comparison
    QPSK_Binary_Theoretical_BER(i) = (0.5)*erfc(sqrt(1/QPSK_Binary_Noise_Power_Spectral_Density(i)));
end

%%%%%%%%%%%%%% plotting BER of QPSK %%%%%%%%%%%%%%%%%
figure(4)
semilogy(SNR_dB,QPSK_Binary_BER , '-','linewidth',2 ) ;
hold on
semilogy( SNR_dB, QPSK_Binary_Theoretical_BER ,'-','linewidth',2) ;
xlabel('Eb/No');
ylabel('BER');
legend('QPSK BER' , 'Theoretical BER ') ;
grid on
title('QPSK Binary-Coded Modulation');

% Plot comparing BER of Gray and Binary coded QPSK
figure(5)
semilogy(SNR_dB,QPSK_BER,'-','linewidth',2)  % Gray coded BER (already plotted previously)
hold on
semilogy(SNR_dB,QPSK_Binary_Theoretical_BER,'-','linewidth',2)  % Binary coded theoretical BER
hold on
semilogy(SNR_dB,QPSK_Binary_BER,'-','linewidth',2)  % Binary coded simulated BER
legend('Simulated BER (Gray)', 'Theoretical BER (Gray)', 'Simulated BER (Binary)');
grid on
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('SNRdB Vs BER plot for QPSK Modulation');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 8PSK Modulation Scheme
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Mapper %%%%%%%%%%%%%%%%%%%%%%%%
% Initialize PSK8_Mapped_bits vector
PSK8_Mapped_bits = zeros(1, (Bits_number - 1) / 3);

% Map each group of 3 bits to corresponding 8-PSK symbols
for i = 1 : (Bits_number - 1) / 3
    current_bits = Random_bits(i * 3 - 2 : i * 3); % Extract current 3 bits
    % Map binary bits to complex symbols based on 8-PSK mapping
    if isequal(current_bits, [0 0 0])
        PSK8_Mapped_bits(i) = cos(0) + 1i * sin(0);
    elseif isequal(current_bits, [0 0 1])
        PSK8_Mapped_bits(i) = cos(pi/4) + 1i * sin(pi/4);
    elseif isequal(current_bits, [0 1 1])
        PSK8_Mapped_bits(i) = cos(pi/2) + 1i * sin(pi/2);
    elseif isequal(current_bits, [0 1 0])
        PSK8_Mapped_bits(i) = cos(3*pi/4) + 1i * sin(3*pi/4);
    elseif isequal(current_bits, [1 1 0])
        PSK8_Mapped_bits(i) = cos(pi) + 1i * sin(pi);
    elseif isequal(current_bits, [1 1 1])
        PSK8_Mapped_bits(i) = cos(5*pi/4) + 1i * sin(5*pi/4);
    elseif isequal(current_bits, [1 0 1])
        PSK8_Mapped_bits(i) = cos(3*pi/2) + 1i * sin(3*pi/2);
    elseif isequal(current_bits, [1 0 0])
        PSK8_Mapped_bits(i) = cos(7*pi/4) + 1i * sin(7*pi/4);
    end
end

%%%%%%%%%%%%%%%%%%%% channel %%%%%%%%%%%%%%%%%%
% Calculate noise power spectral density based on SNR
PSK8_Bit_energy=1/3;
PSK8_Noise_Power_Spectral_Density = PSK8_Bit_energy./(10.^(SNR_dB/10));

% Initialize BER vectors
PSK8_BER = zeros(1, length(SNR_dB));
PSK8_Theoretical_BER = zeros(1, length(SNR_dB));

% Loop over each SNR value
for i = 1 : length(SNR_dB)
    % Generate AWGN with specified power spectral density
    PSK8_AWGN = randn(1,(Bits_number-1)/3)*sqrt(PSK8_Noise_Power_Spectral_Density(i)/2)+ 1i.*randn(1,(Bits_number-1)/3)*sqrt(PSK8_Noise_Power_Spectral_Density(i)/2);
    % Add noise to transmitted symbols to simulate channel
    PSK8_Received_signal = PSK8_Mapped_bits + PSK8_AWGN ;

    %%%%%%%%%%%%%%%%%% Demapper %%%%%%%%%%%%%%%%
    % Attempt to demap received symbols back to bits
    PSK8_Demapped_signal = zeros(1, Bits_number-1);
    for k = 1 : (Bits_number - 1) / 3
        current_angle = angle(PSK8_Received_signal(k));
        % Determine which symbol corresponds to which bits based on angle
        if current_angle >= -pi/8 && current_angle <= pi/8
            PSK8_Demapped_signal((k-1)*3+1 : (k-1)*3+3) = [0, 0, 0];
        elseif current_angle >= pi/8 && current_angle <= 3*pi/8
            PSK8_Demapped_signal((k-1)*3+1 : (k-1)*3+3) = [0, 0, 1];
        elseif current_angle >= 3*pi/8 && current_angle <= 5*pi/8
            PSK8_Demapped_signal((k-1)*3+1 : (k-1)*3+3) = [0, 1, 1];
        elseif current_angle >= 5*pi/8 && current_angle <= 7*pi/8
            PSK8_Demapped_signal((k-1)*3+1 : (k-1)*3+3) = [0, 1, 0];
        elseif current_angle >= -7*pi/8 && current_angle <= -5*pi/8
            PSK8_Demapped_signal((k-1)*3+1 : (k-1)*3+3) = [1, 1, 1];
        elseif current_angle >= -5*pi/8 && current_angle <= -3*pi/8
            PSK8_Demapped_signal((k-1)*3+1 : (k-1)*3+3) = [1, 0, 1];
        elseif current_angle >= -3*pi/8 && current_angle <= -pi/8
            PSK8_Demapped_signal((k-1)*3+1 : (k-1)*3+3) = [1, 0, 0];
        else
            PSK8_Demapped_signal((k-1)*3+1 : (k-1)*3+3) = [1, 1, 0];
        end
    end

    %%%%%%%%%%%%%%%%%%%%%5 BER calculation %%%%%%%%%%%%%%%%
    % Compute Bit Error Rate (BER)
    PSK8_Error = abs( PSK8_Demapped_signal - Random_bits( 1 : (Bits_number-1) ) );
    PSK8_BER(i) = sum(PSK8_Error) / Bits_number;

    % Calculate theoretical BER
    PSK8_Theoretical_BER(i) = (1/3) * erfc(sqrt(1/PSK8_Noise_Power_Spectral_Density(i)) * sin(pi/8));

end

% plotting BER of 8PSK
figure(2)
semilogy(SNR_dB,PSK8_BER , '-','linewidth',2 ) ;
hold on
semilogy( SNR_dB, PSK8_Theoretical_BER ,'-','linewidth',2) ;
xlabel('Eb/No');
ylabel('BER');
legend('8PSK BER' , 'Theoretical BER ');
grid on
title('8PSK Modulation');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 16QAM Modulation Scheme
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Preallocate QAM16_Mapped_bits and QAM16_Demapped_bits arrays
QAM16_Mapped_bits = zeros(1, Bits_number / 4);
QAM16_Demapped_bits = zeros(1, Bits_number);

% Define constellation points
constellation_points = [-3-3j, -3-1j, -3+3j, -3+1j, ...
    -1-3j, -1-1j, -1+3j, -1+1j, ...
    3-3j,  3-1j,  3+3j,  3+1j, ...
    1-3j,  1-1j,  1+3j,  1+1j];

% Map bits to constellation points
for i = 1:Bits_number/4
    index = bi2de(Random_bits(4*i-3 : 4*i)) + 1;
    QAM16_Mapped_bits(i) = constellation_points(index);
end

% Channel simulation
QAM16_Bit_energy = 2.5;
QAM16_Noise_Power_Spectral_Density = QAM16_Bit_energy ./ (10.^(SNR_dB/10));
BER_16QAM = zeros(1, length(SNR_dB));
Theoretical_BER_16QAM = (3/8) * erfc(sqrt(1./QAM16_Noise_Power_Spectral_Density));

for i = 1:length(SNR_dB)
    % Add AWGN
    AWGN_16QAM = randn(1, Bits_number/4) * sqrt(QAM16_Noise_Power_Spectral_Density(i)/2) + ...
        1j * randn(1, Bits_number/4) * sqrt(QAM16_Noise_Power_Spectral_Density(i)/2);
    QAM16_Recieved_signal = QAM16_Mapped_bits + AWGN_16QAM;

    % Demapping bits
    for k = 1:Bits_number/4
        distances = abs(QAM16_Recieved_signal(k) - constellation_points);
        [~, idx] = min(distances);
        demapped_bits = de2bi(idx - 1, 4);
        QAM16_Demapped_bits(4*(k-1) + 1 : 4*k) = demapped_bits;
    end

    % Calculate Bit Error Rate (BER)
    error_bits = abs(QAM16_Demapped_bits - Random_bits);
    BER_16QAM(i) = sum(error_bits) / Bits_number;
end

% Plotting BER
figure(6);
semilogy(SNR_dB, BER_16QAM, '-', 'linewidth', 2);
hold on;
semilogy(SNR_dB, Theoretical_BER_16QAM, '-', 'linewidth', 2);
xlabel('Eb/No');
ylabel('BER');
legend('16QAM BER', 'Theoretical BER');
grid on;
title('16QAM Modulation');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BFSK Modulation Scheme
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%------------------------- BFSK -----------------------------
%%%%%%%%%%%%%%%%%%%%%%%% Mapper %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BFSK = zeros(1 , Bits_number) ;
for i = 1 : (Bits_number)
    if Random_bits(i)  == 0
        BFSK(i) = cos(0)+1i*sin(0);
    else
        BFSK(i) = cos(pi/2)+1i*sin(pi/2);
    end
end
%%%%%%%%%%%%%%%%%%%% channel %%%%%%%%%%%%%%%%%%
BFSK_Bit_energy=1;
BFSK_Noise_Power_Spectral_Density = BFSK_Bit_energy./(10.^(SNR_dB/10));
BER_BFSK = zeros(1, length(SNR_dB));
Theoretical_BER_BFSK = zeros(1, length(SNR_dB));

% Main loop
for i = 1:length(SNR_dB)
    % Generate AWGN
    BFSK_AWGN = randn(1, Bits_number) * sqrt(BFSK_Noise_Power_Spectral_Density(i) / 2) + 1i .* randn(1, Bits_number) * sqrt(BFSK_Noise_Power_Spectral_Density(i) / 2);

    % Add noise to transmitted signal
    BFSK_Received_signal = BFSK + BFSK_AWGN;

    % Demapper
    demapped_BFSK = zeros(1, Bits_number);
    for k = 1:Bits_number
        if angle(BFSK_Received_signal(k)) >= -3*pi/4 && angle(BFSK_Received_signal(k)) < pi/4
            demapped_BFSK(k) = 0;
        else
            demapped_BFSK(k) = 1;
        end
    end

    % Calculate BER
    BFSK_Error = abs(demapped_BFSK - Random_bits(1:Bits_number));
    BER_BFSK(i) = sum(BFSK_Error) / Bits_number;

    % Calculate Theoretical BER
    Theoretical_BER_BFSK(i) = (1/2) * erfc(sqrt(0.5 / BFSK_Noise_Power_Spectral_Density(i)));
end
%%%%%%%%%%%%%%%%% plotting BER of BFSK %%%%%%%%%%%%%%%%%%%
figure(7)
semilogy(SNR_dB,BER_BFSK , '-','linewidth',2 ) ;
hold on
semilogy( SNR_dB, Theoretical_BER_BFSK ,'-','linewidth',2) ;
xlabel('Eb/No');
ylabel('BER');
legend('BFSK BER' , 'Theoretical BER ') ;
grid on
title('BFSK Modulation');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The modulation schemes under consideration: BPSK, QPSK, 8PSK, BFSK and 16QAM systems
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(8)
semilogy(SNR_dB,QPSK_BER,'-','linewidth',2)
hold on
semilogy(SNR_dB,BPSK_BER,'-','linewidth',2)
hold on
semilogy(SNR_dB,BER_16QAM,'-','linewidth',2)
hold on
semilogy(SNR_dB,PSK8_BER,'-','linewidth',2)
hold on
semilogy(SNR_dB,BER_BFSK,'-','linewidth',2)
hold on
semilogy(SNR_dB,QPSK_Theoretical_BER,'--','linewidth',2)
hold on
semilogy(SNR_dB,BPSK_Theoretical_BER,'--','linewidth',2)
hold on
semilogy(SNR_dB,Theoretical_BER_16QAM,'--','linewidth',2)
hold on
semilogy(SNR_dB,PSK8_Theoretical_BER,'--','linewidth',2)
hold on
semilogy(SNR_dB,Theoretical_BER_BFSK,'--','linewidth',2)

legend(' BER QPSK(gray)', ' BPSK BER ',' 16QAM BER ',' 8PSK BER',' BFSK BER','theoretical QPSK(Gray) ','theoretical BPSK ','theoretical 16QAM','theoretical 8PSK','theoretical BFSK');
grid on
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('SNR Vs BER plot for different Modualtion schemes');
ylim([1e-3 1]);
xlim([-2 5]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BFSK PSD Section 1.5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Number of realizations
num_realizations = 500;
% Generate random data (0s and 1s)
data = randi([0, 1], num_realizations, 101);
% Repeat data for symbol transmission
data_repeated = repelem(data, 1, 7);
% Bit period (in samples)
bit_period = 7;
% Time samples within a bit period
time_samples = 0:1:bit_period - 1;
% Energy per bit
energy_per_bit = 1;
% Random delay for each realization
delay = randi([1, bit_period], 1, num_realizations);
% Complex baseband equivalent for '0'
symbol_0 = sqrt(2 * energy_per_bit / bit_period);
% Complex baseband equivalent for '1'
symbol_1 = symbol_0 * (cos(2 * pi * time_samples / bit_period) + 1j * sin(2 * pi * time_samples / bit_period));
% Create transmitted signal for each realization
for i = 1:num_realizations
    symbol_index = 1;
    for j = 1:101
        if data(i, j) == 1
            data_repeated(i, symbol_index:symbol_index + 6) = symbol_1;
        else
            data_repeated(i, symbol_index:symbol_index + 6) = symbol_0;
        end
        symbol_index = symbol_index + 7;
    end
end
% Introduce delay in transmitted signal
transmitted_signal = zeros(num_realizations, 700);
for i = 1:num_realizations
    transmitted_signal(i, :) = [data_repeated(i, 700 - delay(i) + 1:700) data_repeated(i, 1:700 - delay(i))];
end
transmitted_signal = transmitted_signal(1:num_realizations, 1:700);
% Number of samples for ACF calculation
num_samples = 7;
% Initialize ensemble ACF
ensemble_ACF(1, 7 * 100) = 0;
% Calculate ensemble ACF
for i = 1:700
    for j = 1:num_realizations
        ensemble_ACF(i, j) = sum(conj(transmitted_signal(:, i)) .* transmitted_signal(:, j)) / num_realizations;
    end
end
% Create complete ACF for plotting
ensemble_acf_complete = [conj(fliplr(ensemble_ACF(1, :))) ensemble_ACF(1, :)];
%%%%%%%%%%%%%%%%%%%%%% plotting the PSD %%%%%%%%%%%%%%%
figure(9)
acf_frequencies = (-700:699) / 1400;
psd = fftshift(fft(ensemble_acf_complete(1, :)));
plot(acf_frequencies, abs(psd) / 1400)
title('Power Spectral Density (PSD) of BFSK')
ylabel('S(f)');
xlabel('Frequency');