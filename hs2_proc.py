function CAL = hs2_read_cal_file(cal_filename);

% Date Created:  June 21, 2007
% Date Modified: June 26, 2007
%
% Brandon Sackmann
% Postdoctoral Fellow
% Monterey Bay Aquarium Research Institute
% 7700 Sandholdt Road
% Moss Landing, California  95039
%
% Tel: (831) 775-1958
% Fax: (831) 775-1620
% Email: sackmann@mbari.org


% Open either a URL or a file
try
        url = java.net.URL(cal_filename);
catch
        file = java.io.File(cal_filename);
        url = file.toURL;
end

is = url.openStream;
isr = java.io.InputStreamReader(is);
br = java.io.BufferedReader(isr);

CAL = [];
channel = 0;
while 1
    
    str = br.readLine;

    % Convert java String to matlab String
    S = deblank(char(str));
 
    if ~isempty(strfind(S,'General'))
        category = 'General';
    	str = br.readLine; S = deblank(char(str));
    elseif ~isempty(strfind(S,'Channel'))
        channel = channel+1;
        category = ['Ch(' num2str(channel) ')'];
    	str = br.readLine; S = deblank(char(str));
    elseif ~isempty(strfind(S,'End'))
        break
    end
    
    if ~isempty(S)
        CAL = calparse(CAL,S,category);
    end
end


% NB:  The range of values for SigmaExp is relatively small and the error
%      in the final output that results from our choice of SigmaExp (when a
%      value is not provided in the calibration file) is likely to be <1%
%      Value needs to be a string as hs2_calc_bb calls str2num() on it.
if ~isfield(CAL.Ch,'SigmaExp') & strcmp(CAL.General.Serial,'H2000325')
    CAL.Ch(1).SigmaExp = '0.1460';     % value obtained from subsequent calibrations of this instrument
    CAL.Ch(2).SigmaExp = '0.1600';     % value obtained from subsequent calibrations of this instrument
elseif ~isfield(CAL.Ch,'SigmaExp')
    CAL.Ch(1).SigmaExp = '0.1486';     % average SigmaExp value for sensors H2000325 and H2D021004 from 2001-2007 [(0.153+0.145+0.153+0.146+0.146)/5]
    CAL.Ch(2).SigmaExp = '0.1522';     % average SigmaExp value for sensors H2000325 and H2D021004 from 2001-2007 [(0.150+0.142+0.150+0.159+0.160)/5]
end

function CAL = calparse(CAL,S,category);
    [T,R] = strtok(S,'=');
    eval(['CAL.' category '.' T '= char(R(2:end));'])


def _hs2_calc_bb(hs2,CAL) {

    # Translated from hs2_calc_bb.m - preserving original comments 
    # % Date Created:  June 21, 2007
    # % Date Modified: June 26, 2007
    # %
    # % Brandon Sackmann
    # % Postdoctoral Fellow
    # % Monterey Bay Aquarium Research Institute
    # % 7700 Sandholdt Road
    # % Moss Landing, California  95039
    # %
    # % Tel: (831) 775-1958
    # % Fax: (831) 775-1620
    # % Email: sackmann@mbari.org
    # 
    # % FIND REAL GAIN NUMBER FROM CAL FILE AND HS2 POINTERS
    for channel in (1, 2, 3):
        for gain in (1, 2, 3, 4, 5):
            eval(['ind=find(hs2.Gain' num2str(channel) '==' num2str(gain) ');']);
            if channel <= 2
                eval(['hs2.Gain' num2str(channel) '(ind)=str2num(CAL.Ch(' num2str(channel) ').Gain' num2str(gain) ');']);
            elseif channel == 3
                eval(['hs2.Gain' num2str(channel) '(ind)=str2num(CAL.Ch(' num2str(channel-1) ').Gain' num2str(gain) ');']);
            end
        end
    end

% RAW SIGNAL CONVERSION
eval(['hs2.beta' CAL.Ch(1).Name(3:end) '_uncorr = (hs2.Snorm1.*str2num(CAL.Ch(1).Mu))./((1 + str2num(CAL.Ch(1).TempCoeff).*(hs2.Temp-str2num(CAL.General.CalTemp))).*hs2.Gain1.*str2num(CAL.Ch(1).RNominal));'])
eval(['hs2.beta' CAL.Ch(2).Name(3:end) '_uncorr = (hs2.Snorm2.*str2num(CAL.Ch(2).Mu))./((1 + str2num(CAL.Ch(2).TempCoeff).*(hs2.Temp-str2num(CAL.General.CalTemp))).*hs2.Gain2.*str2num(CAL.Ch(2).RNominal));'])
eval(['hs2.fl' CAL.Ch(3).Name(3:end) '_uncorr = (hs2.Snorm3.*50)./((1 + str2num(CAL.Ch(3).TempCoeff).*(hs2.Temp-str2num(CAL.General.CalTemp))).*hs2.Gain3.*str2num(CAL.Ch(3).RNominal));'])
hs2.caldepth = (str2num(CAL.General.DepthCal).*hs2.Depth) - str2num(CAL.General.DepthOff);

for channel = 1:2
    eval(['beta_uncorr = hs2.beta' CAL.Ch(channel).Name(3:end) '_uncorr;']) 
    
    % BACKSCATTERING COEFFICIENT CALCULATION
    [beta_w, b_bw]  = purewater_scatter(str2num(CAL.Ch(channel).Name(3:end)));
    chi             = 1.08;

    b_b_uncorr      = ((2*pi*chi).*(beta_uncorr - beta_w)) + b_bw;

    eval(['hs2.bb' CAL.Ch(channel).Name(3:end) '_uncorr = b_b_uncorr;']) 
    eval(['hs2.bbp' CAL.Ch(channel).Name(3:end) '_uncorr = b_b_uncorr - b_bw;']) 

    % ESTIMATION OF KBB AND SIGMA FUNCTION
    a           =   typ_absorption(str2num(CAL.Ch(channel).Name(3:end)));
    b_b_tilde   =   0.015;
    b           =   (b_b_uncorr - b_bw)./b_b_tilde;

    K_bb        =   a + 0.4.*b;
    k_1         =   1.0;
    k_exp       =   str2num(CAL.Ch(channel).SigmaExp);
    sigma       =   k_1.*exp(k_exp.*K_bb);

    b_b_corr    =   sigma.*b_b_uncorr;
    
    eval(['hs2.bb' CAL.Ch(channel).Name(3:end) ' = b_b_corr;']) 
    eval(['hs2.bbp' CAL.Ch(channel).Name(3:end) ' = b_b_corr - b_bw;']) 
end

    
% ***********************************
% SUBFUNCTION:  PURE-WATER SCATTERING
% ***********************************
function [beta_w, b_bw] = purewater_scatter(lamda);

% assumes lamda is a scalar

beta_w_ref  =   2.18E-04;   % for seawater
b_bw_ref    =   1.17E-03;   % for seawater
%beta_w_ref  =   1.67E-04;   % for freshwater
%b_bw_ref    =   8.99E-04;   % for freshwater
lamda_ref   =   525;
gamma       =   4.32;

beta_w      =   beta_w_ref*(lamda_ref/lamda)^gamma;
b_bw        =   b_bw_ref*(lamda_ref/lamda)^gamma;


% **********************************
% SUBFUNCTION:  'TYPICAL' ABSORPTION
% **********************************
function [a] = typ_absorption(lamda);

% assumes lamda is a scalar

C           =   0.1;
gamma_y     =   0.014;
a_d_400     =   0.01;
gamma_d     =   0.011;


% Embed the lookup table from the AStar.CSV file here
%%a_star    =   load('AStar.csv');
a_star      = [ 400,0.687;...
		410,0.828;...
		420,0.913;...
		430,0.973;...
		440,1;...
		450,0.944;...
		460,0.917;...
		470,0.87;...
		480,0.798;...
		490,0.75;...
		500,0.668;...
		510,0.618;...
		520,0.528;...
		530,0.474;...
		540,0.416;...
		550,0.357;...
		560,0.294;...
		570,0.276;...
		580,0.291;...
		590,0.282;...
		600,0.236;...
		610,0.252;...
		620,0.276;...
		630,0.317;...
		640,0.334;...
		650,0.356;...
		660,0.441;...
		670,0.595;...
		680,0.502;...
		690,0.329;...
		700,0.215;...
              ];

a_star      =   interp1(a_star(:,1),a_star(:,2),lamda);

a           =   (0.06*a_star*(C^0.65))*(1+0.2*exp(-gamma_y*(lamda-440))) + ...
                (a_d_400*exp(-gamma_d*(lamda-400)));
