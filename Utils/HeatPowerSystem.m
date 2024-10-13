function mpc = HeatPowerSystem
%CASE5  Power flow data for modified 5 bus, 5 gen case based on PJM 5-bus system
%   Please see CASEFORMAT for details on the case file format.
%
%   Based on data from ...
%     F.Li and R.Bo, "Small Test Systems for Power System Economic Studies",
%     Proceedings of the 2010 IEEE Power & Energy Society General Meeting

%   Created by Rui Bo in 2006, modified in 2010, 2014.
%   Distributed with permission.

%   MATPOWER
%   $Id: case5.m 2324 2014-05-23 18:01:41Z ray $

%% MATPOWER Case Format : Version 2
mpc.version = '2';

%%-----  Power Flow Data  -----%%
%% system MVA base
mpc.baseMVA = 100;

%% bus data
%	bus_i type  Pd  Qd	 Gs	Bs	area  Vm	Va	baseKV	zone Vmax Vmin
mpc.bus = [
    1	3	0	0	0	0	1	1.05	0	230	1	1.05	0.95;
    2	2	0	0	0	0	1	1.05	0	230	1	1.05	0.95;
    3	2	50	0	0	0	1	1.07	0	230	1	1.05	0.95;
    4	1	50	0	0	0	1	1	    0	230	1	1.05	0.95;
    5	1	50	0	0	0	1	1       0	230	1	1.05	0.95;
    6	1	0	0	0	0	1	1       0	230	1	1.05	0.95;

];

%% branch data
%  fbus	tbus r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
    1	2	0	0.17	0	400	200	200	0	0	1	-360	360;
    1	4	0	0.258	0	200	100	100	0	0	1	-360	360;
    2	3	0	0.197	0	200	100	100	0	0	1	-360	360;
    2	4	0	0.018	0	200	100	100	0	0	1	-360	360;
    3	6	0	0.037	0	800	100	100	0	0	1	-360	360;
    4	5	0	0.037	0	200	100	100	0	0	1	-360	360;
    5	6	0	0.14	0	200	100	100	0	0	1	-360	360;
];




%% generator data
%	bus	Pg	Qg	Qmax  Qmin	Vg     mBase	status	Pmax   Pmin minup mindown  initup initdown	rampup	rampdown	type
mpc.gen = [
    1	0	0	100	 -80	1.05	100      1       120     30	  4      4       1       0       30      30	    1
    5	0	0	70	 -40	1.05	100      1       100     20	  3      2       0       1       30      30	    1
    6	0	0	150	 -50	1	    100      1       100     20	  2      1       0       1       50      50     2
];
% Generator at bus 1 is changed to CHP

%%-----  OPF Data  -----%%
%% generator cost data
%	1	startup	shutdown	n	x1	y1	...	xn	yn
%	2	startup	shutdown	n	c(n-1)	...	c0
mpc.gencost = [
    2	124.69	0	3	0.0005	17	220
    2	373.83	0	3	0.0012	30	100
    2	100     0	3	0.0060	26	171
];

%% Wind Farm
%       bus  Pmax
mpc.WF=[ 2  150    
         3  150
       ];
% 24-hour wind power forecast value (MW)
mpc.WFforecast=[
111	104
108	103
106	101
103	 97
94	 82
92	 72
79   65
70	 55
68	 50
59	 49
54	 48
51	 47
49	 51
48	 55
49	 68
54	 75
68	 81
78	 86
85	 93
90	 96
97	 100
100	 108
102  111
105	 112
]';

%% Hydrogen Storage System
%        PowerBUS HeatBUS  Pmax (MW)  ELZ_eta  ELZ_Hex_eta   
mpc.ELZ=[ 2        2        50          0.7        0.8  
          3        1        50          0.7        0.8  
];

%        bus  Hmax (kg)  H_charge(kg/h)  H_discharge(kg/h)  H_initial(kg)  
mpc.HT=[ 2    1800          1000            1000                200
         3    1800          1000            1000                200
];

%        PowerBUS  HeatBUS   Pmax (MW)  FC_eta  FC_Hex_eta   
mpc.FC=[    2         2       50          0.3        0.8  
            3         1       50          0.3        0.8 
];


%% Active Load
mpc.load = [
    184
    194
    193
    195
    197
    200
    213
    228
    235
    237
    256
    258
    255
    253
    254
    231
    210
    202
    207
    196
    199
    200
    198
    203
]';


%% Heat System
mpc.HeatBranch = [
%index fbus tbus FlowWater(kg/s) length(m)  diameter(m) roughness conductivity
    1   1   2   265.85          2000        0.8         0.0005	    0.12;
    2   2   3   265.85          2000        0.8         0.0005	    0.12;
    3   3   4   241.41          1750        0.8         0.0005	    0.12;
    4   4   5   143.58          1750        0.8         0.0005	    0.12;
    5   3   6    24.44          1750        0.8         0.0005	    0.12;
    6   4   7    97.83          1750        0.8         0.0005	    0.12;
];

mpc.HeatBranch(:,4) = mpc.HeatBranch(:,4)*3600; %(kg/h)

mpc.HeatBus = [
% index busType %(3=热源，2=负荷，1=不发生热交换的节点)   Load    Tsmin   Tsmax   Trmin   Trmax
    1       3                                           0       110     120     60      80;  
    2       1                                           0       110     120     60      80;
    3       1                                           0       110     120     60      80;
    4       1                                           0       110     120     60      80;
    5       2                                           47.2    110     120     60      80;
    6       2                                           8       110     120     60      80;
    7       2                                           32      110     120     60      80;
];

mpc.SituationTempreture = [
    -10 -10 -8.84   -9.42   -9.42   -9.42   -8.84   -8.26   -7.10   -6.52   -5.94   -5.35   -4.77   -4.77   -4.77   -5.35   -5.94   -6.52   -6.52   -6.52   -7.10   -7.68   -8.26   -8.26
];

% Feasible operating region of CHP; Convex Polyhedron
mpc.CHPgen=[
%  HeatBUS PowerBUS   Npoint   p1	h1	p2	 h2	   p3	h3	 p4	    h4
   1         6          4      40	0	38	 34	   78	60	 90	 0   
];

mpc.CHPcost=[
%  HeatBUS PowerBUS  q^1      q^2       q*p
   1         6       0.06     0.0027    0.0041            
];


mpc.HeatLoad = [
    54.945	55  52.954	54.01	54.021	54.043	52.932	51.942	49.709	48.62	47.575	46.486	45.496	45.518	45.551	46.508	47.663	48.664	48.664	48.697	49.786	50.776	51.909	51.92   
];
end
