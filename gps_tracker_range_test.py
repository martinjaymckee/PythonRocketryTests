import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

debug_text = """
0,	6,	6,	-92,	0,	2250,	2257,	37.999967974,	-104.000046857,	37.999964331,	-103.999986986,	0x5E11
6,	11,	13,	-68,	5000000,	2241,	2255,	38.000097259,	-103.999870692,	38.000031716,	-104.000016451,	0xD195
12,	7,	11,	-35,	10000000,	2249,	2250,	38.000016000,	-103.999905069,	38.000068221,	-104.000076796,	0x6C82
18,	12,	9,	-2,	15000000,	2252,	2242,	38.000083525,	-103.999790664,	38.000073846,	-103.999988357,	0x5DF7
0,	8,	7,	-95,	20000000,	2253,	2250,	38.000120171,	-103.999707137,	38.000048592,	-103.999930798,	0xBD69
6,	13,	14,	-69,	25000000,	2251,	2254,	38.000125936,	-103.999474827,	37.999992458,	-104.000083781,	0x30C9
12,	9,	12,	-38,	30000000,	2246,	2255,	38.000280486,	-103.999453060,	38.000085108,	-104.000087981,	0x615C
18,	14,	10,	4,	35000000,	2255,	2253,	38.000224492,	-103.999282510,	37.999967215,	-103.999943397,	0x983D
0,	10,	8,	-93,	40000000,	2243,	2248,	38.000317282,	-103.999322502,	37.999998106,	-104.000009357,	0x9254
6,	6,	6,	-65,	45000000,	2247,	2258,	38.000379193,	-103.999213711,	37.999998117,	-103.999926532,	0x18FE
12,	11,	13,	-35,	50000000,	2247,	2246,	38.000410224,	-103.999135799,	37.999967248,	-104.000054250,	0xD210
18,	7,	11,	1,	55000000,	2244,	2250,	38.000410375,	-103.999088767,	38.000085163,	-104.000033185,	0xE621
0,	12,	9,	-101,	60000000,	2256,	2250,	38.000379647,	-103.998892951,	37.999992535,	-104.000043000,	0xEF24
6,	8,	7,	-71,	65000000,	2247,	2247,	38.000497702,	-103.998907678,	38.000048691,	-104.000083693,	0xFE13
12,	13,	14,	-28,	70000000,	2253,	2241,	38.000584878,	-103.998773621,	38.000073967,	-103.999975604,	0xEBFD
18,	9,	12,	-10,	75000000,	2256,	2250,	38.000461511,	-103.998670445,	38.000068363,	-104.000078057,	0x8EF7
0,	14,	10,	-90,	80000000,	2255,	2257,	38.000486927,	-103.998598147,	38.000031880,	-104.000031727,	0x1B7B
6,	10,	8,	-72,	85000000,	2252,	2241,	38.000661127,	-103.998556730,	37.999964518,	-104.000016276,	0xFEE3
12,	6,	6,	-31,	90000000,	2245,	2241,	38.000624784,	-103.998366528,	38.000045938,	-104.000031705,	0x7FF4
18,	11,	13,	-3,	95000000,	2254,	2256,	38.000737225,	-103.998207207,	37.999916816,	-104.000078013,	0x31B9
0,	7,	11,	-102,	100000000,	2241,	2250,	38.000639123,	-103.998258428,	37.999936478,	-103.999975538,	0x4CF1
6,	12,	9,	-68,	105000000,	2243,	2259,	38.000689805,	-103.998160865,	37.999925260,	-104.000083606,	0x6FC5
12,	8,	7,	-42,	110000000,	2242,	2246,	38.000709607,	-103.998094183,	38.000062826,	-104.000042890,	0xBC11
18,	13,	14,	-6,	115000000,	2257,	2249,	38.000878192,	-103.997878716,	37.999989848,	-104.000033054,	0xEFBA
0,	9,	12,	-96,	120000000,	2250,	2248,	38.000836235,	-103.997873793,	38.000065655,	-104.000054097,	0x03B1
6,	14,	10,	-74,	125000000,	2257,	2244,	38.000943061,	-103.997720085,	37.999930918,	-103.999926357,	0x9D03
12,	10,	8,	-36,	130000000,	2244,	2255,	38.001019008,	-103.997597258,	37.999944966,	-104.000009159,	0x184D
18,	6,	6,	-18,	135000000,	2245,	2245,	38.000884412,	-103.997505310,	37.999928133,	-103.999943178,	0x974D
0,	11,	13,	-99,	140000000,	2244,	2250,	38.001078263,	-103.997444242,	38.000060084,	-104.000087740,	0x50A2
6,	7,	11,	-75,	145000000,	2258,	2251,	38.001061571,	-103.997414053,	37.999981493,	-104.000083518,	0xC4F0
12,	12,	9,	-38,	150000000,	2250,	2250,	38.001013999,	-103.997235080,	38.000051684,	-103.999930513,	0x30B5
18,	8,	7,	-11,	155000000,	2257,	2246,	38.001115211,	-103.997266651,	37.999911334,	-103.999988050,	0x387B
0,	13,	14,	-98,	160000000,	2243,	2257,	38.001185543,	-103.997149438,	37.999919766,	-104.000076467,	0xB549
6,	9,	12,	-72,	165000000,	2244,	2246,	38.001224996,	-103.997063104,	38.000076982,	-104.000016100,	0x3621
12,	14,	10,	-36,	170000000,	2242,	2250,	38.001233569,	-103.997007650,	38.000023656,	-103.999986614,	0x366D
18,	10,	8,	-14,	175000000,	2255,	2251,	38.001211262,	-103.996803413,	37.999939450,	-103.999988006,	0xAFE8
0,	6,	6,	-92,	180000000,	2247,	2250,	38.001337739,	-103.996809718,	38.000004027,	-104.000020278,	0x1F72
6,	11,	13,	-64,	185000000,	2253,	2245,	38.001253673,	-103.996667240,	38.000037725,	-104.000083430,	0xD5BE
12,	7,	11,	-44,	190000000,	2257,	2255,	38.001318391,	-103.996555642,	38.000040543,	-103.999997799,	0x8801
18,	12,	9,	-13,	195000000,	2258,	2244,	38.001352229,	-103.996474923,	38.000012482,	-103.999943047,	0xC603
0,	8,	7,	-96,	200000000,	2255,	2248,	38.001355188,	-103.996425083,	37.999953541,	-103.999919174,	0x5342
6,	13,	14,	-65,	205000000,	2250,	2249,	38.001506930,	-103.996406123,	38.000043383,	-103.999926181,	0x1013
12,	9,	12,	-47,	210000000,	2259,	2246,	38.001448129,	-103.996238380,	37.999922683,	-103.999964068,	0x3AB8
18,	14,	10,	-21,	215000000,	2247,	2259,	38.001538112,	-103.996101516,	37.999950766,	-104.000032834,	0xE4B5
0,	10,	8,	-95,	220000000,	2250,	2250,	38.001597216,	-103.995995532,	37.999947970,	-103.999952817,	0x0AA8
6,	6,	6,	-76,	225000000,	2250,	2256,	38.001625439,	-103.995920428,	37.999914294,	-104.000083343,	0x75F3
12,	11,	13,	-45,	230000000,	2246,	2241,	38.001622783,	-103.995876203,	38.000029402,	-104.000065085,	0xEB02
18,	7,	11,	-24,	235000000,	2258,	2241,	38.001768911,	-103.995862857,	37.999933967,	-104.000077706,	0xED43
0,	12,	9,	-103,	240000000,	2249,	2257,	38.001704496,	-103.995700728,	37.999987315,	-103.999941544,	0x04B5
6,	8,	7,	-68,	245000000,	2254,	2250,	38.001788864,	-103.995569479,	38.000009784,	-104.000015925,	0x96AE
12,	13,	14,	-53,	250000000,	2257,	2241,	38.001842353,	-103.995469109,	38.000001373,	-103.999941522,	0xD190
18,	9,	12,	-22,	255000000,	2256,	2247,	38.001864962,	-103.995399619,	37.999962083,	-104.000077662,	0x42A1
0,	14,	10,	-93,	260000000,	2252,	2250,	38.001856692,	-103.995361009,	38.000071576,	-104.000065019,	0xCA07
6,	10,	8,	-70,	265000000,	2246,	2250,	38.001817542,	-103.995173615,	37.999970527,	-104.000083255,	0xB31F
12,	6,	6,	-56,	270000000,	2254,	2246,	38.001927175,	-103.995196764,	38.000018261,	-103.999952707,	0x63BC
18,	11,	13,	-30,	275000000,	2259,	2258,	38.002005929,	-103.995071129,	38.000035115,	-104.000032703,	0x050C
0,	7,	11,	-92,	280000000,	2242,	2248,	38.002053804,	-103.994976374,	38.000021090,	-103.999963914,	0xD492
6,	12,	9,	-81,	285000000,	2241,	2253,	38.002070799,	-103.994912498,	37.999976185,	-103.999926006,	0xA736
12,	8,	7,	-54,	290000000,	2255,	2255,	38.002056914,	-103.994879502,	38.000080064,	-103.999918977,	0x09BB
18,	13,	14,	-34,	295000000,	2248,	2254,	38.002191812,	-103.994697723,	37.999973400,	-103.999942827,	0xD11B
0,	9,	12,	-100,	300000000,	2255,	2250,	38.002116168,	-103.994546823,	38.000015519,	-103.999997557,	0x74D3
6,	14,	10,	-73,	305000000,	2241,	2242,	38.002189308,	-103.994606465,	38.000026759,	-104.000083167,	0xB3E8
12,	10,	8,	-48,	310000000,	2243,	2250,	38.002231568,	-103.994517325,	38.000007120,	-104.000019993,	0x7151
18,	6,	6,	-33,	315000000,	2241,	2255,	38.002242948,	-103.994279400,	37.999956600,	-103.999987699,	0x3EEB
0,	11,	13,	-90,	320000000,	2254,	2257,	38.002223449,	-103.994252019,	38.000054864,	-103.999986285,	0x4CD6
6,	7,	11,	-75,	325000000,	2246,	2255,	38.002352733,	-103.994075854,	37.999942586,	-104.000015750,	0x8CC5
12,	12,	9,	-51,	330000000,	2253,	2250,	38.002451137,	-103.994110232,	37.999979091,	-104.000076094,	0xF632
18,	8,	7,	-41,	335000000,	2258,	2242,	38.002338999,	-103.993995826,	37.999984716,	-103.999987655,	0xA1CE
0,	13,	14,	-90,	340000000,	2258,	2250,	38.002375645,	-103.993912299,	37.999959462,	-103.999930096,	0xDF5F
6,	9,	12,	-72,	345000000,	2256,	2254,	38.002561074,	-103.993859653,	38.000082992,	-104.000083079,	0x2148
12,	14,	10,	-64,	350000000,	2250,	2255,	38.002535960,	-103.993658223,	37.999995978,	-104.000087279,	0x8E78
18,	10,	8,	-44,	355000000,	2241,	2253,	38.002479966,	-103.993667335,	38.000057748,	-103.999942696,	0xF364
0,	6,	6,	-99,	360000000,	2248,	2248,	38.002572757,	-103.993527664,	38.000088639,	-104.000008655,	0x6B32
6,	11,	13,	-79,	365000000,	2251,	2258,	38.002634667,	-103.993418873,	38.000088650,	-103.999925830,	0xECC9
12,	7,	11,	-58,	370000000,	2251,	2246,	38.002665698,	-103.993340961,	38.000057781,	-104.000053549,	0x01F1
18,	12,	9,	-43,	375000000,	2249,	2250,	38.002665849,	-103.993293929,	37.999996033,	-104.000032483,	0x1A21
0,	8,	7,	-103,	380000000,	2243,	2250,	38.002814784,	-103.993098113,	38.000083068,	-104.000042298,	0x5AF5
6,	13,	14,	-81,	385000000,	2251,	2247,	38.002753176,	-103.993112840,	37.999959561,	-104.000082992,	0xB6E4
12,	9,	12,	-62,	390000000,	2258,	2241,	38.002840352,	-103.992978784,	37.999984837,	-103.999974902,	0x55F0
18,	14,	10,	-38,	395000000,	2242,	2250,	38.002896648,	-103.992875607,	37.999979234,	-104.000077355,	0x7909
0,	10,	8,	-102,	400000000,	2242,	2257,	38.002922064,	-103.992803309,	37.999942750,	-104.000031025,	0xB6E1
6,	6,	6,	-79,	405000000,	2257,	2241,	38.002916601,	-103.992761892,	38.000055051,	-104.000015574,	0xC60E
12,	11,	13,	-61,	410000000,	2250,	2241,	38.002880259,	-103.992571691,	37.999956809,	-104.000031003,	0x1349
18,	7,	11,	-42,	415000000,	2259,	2256,	38.002992699,	-103.992592032,	38.000007350,	-104.000077311,	0xA601
0,	12,	9,	-97,	420000000,	2246,	2250,	38.003074261,	-103.992463590,	38.000027011,	-103.999974836,	0x01E4
6,	8,	7,	-86,	425000000,	2248,	2259,	38.003124942,	-103.992366028,	38.000015793,	-104.000082904,	0x975B
12,	13,	14,	-69,	430000000,	2247,	2246,	38.003144744,	-103.992299345,	37.999973696,	-104.000042188,	0xB9A6
18,	9,	12,	-41,	435000000,	2243,	2249,	38.003133666,	-103.992263542,	38.000080382,	-104.000032352,	0xF548
0,	14,	10,	-102,	440000000,	2254,	2248,	38.003091709,	-103.992078955,	37.999976525,	-104.000053395,	0x4B69
6,	10,	8,	-74,	445000000,	2243,	2244,	38.003198536,	-103.991925248,	38.000021452,	-103.999925655,	0x8C81
12,	6,	6,	-59,	450000000,	2249,	2255,	38.003274482,	-103.991982083,	38.000035499,	-104.000008457,	0xF0C6
18,	11,	13,	-49,	455000000,	2250,	2245,	38.003319549,	-103.991890135,	38.000018666,	-103.999942476,	0x24C1
0,	7,	11,	-101,	460000000,	2249,	2250,	38.003333737,	-103.991649404,	37.999970954,	-104.000087038,	0xC976
6,	12,	9,	-86,	465000000,	2244,	2251,	38.003317045,	-103.991619215,	38.000072026,	-104.000082816,	0x09A5
12,	8,	7,	-72,	470000000,	2254,	2250,	38.003449136,	-103.991619906,	37.999962555,	-103.999929811,	0x620B
18,	13,	14,	-54,	475000000,	2243,	2246,	38.003370685,	-103.991471813,	38.000001867,	-103.999987348,	0x0EA8
0,	9,	12,	-97,	480000000,	2248,	2257,	38.003441017,	-103.991354600,	38.000010300,	-104.000075765,	0xA530
6,	14,	10,	-79,	485000000,	2249,	2246,	38.003480470,	-103.991268267,	37.999987853,	-104.000015399,	0x630D
12,	10,	8,	-66,	490000000,	2247,	2250,	38.003489043,	-103.991212813,	37.999934526,	-103.999985912,	0x0A1B
18,	6,	6,	-53,	495000000,	2241,	2251,	38.003646399,	-103.991188238,	38.000029983,	-103.999987304,	0xA591
0,	11,	13,	-101,	500000000,	2251,	2250,	38.003593213,	-103.991014881,	37.999914897,	-104.000019577,	0x0E26
6,	7,	11,	-81,	505000000,	2258,	2245,	38.003688811,	-103.990872402,	37.999948595,	-104.000082728,	0xA986
12,	12,	9,	-70,	510000000,	2243,	2255,	38.003753528,	-103.990760804,	37.999951413,	-103.999997097,	0x9235
18,	8,	7,	-62,	515000000,	2244,	2244,	38.003787367,	-103.990680085,	37.999923352,	-103.999942345,	0x7D80
0,	13,	14,	-101,	520000000,	2241,	2248,	38.003790325,	-103.990630245,	38.000044074,	-103.999918472,	0x0483
6,	9,	12,	-79,	525000000,	2254,	2249,	38.003762404,	-103.990611285,	37.999954254,	-103.999925479,	0xC834
12,	14,	10,	-70,	530000000,	2245,	2246,	38.003883267,	-103.990443542,	38.000013216,	-103.999963366,	0x73DC
18,	10,	8,	-66,	535000000,	2251,	2259,	38.003793586,	-103.990306678,	38.000041300,	-104.000032132,	0x5A83
0,	6,	6,	-97,	540000000,	2254,	2250,	38.003852690,	-103.990380357,	38.000038504,	-103.999952115,	0x95A5
6,	11,	13,	-87,	545000000,	2254,	2256,	38.003880913,	-103.990125590,	38.000004828,	-104.000082641,	0xFCC3
12,	7,	11,	-78,	550000000,	2250,	2241,	38.003878257,	-103.990081365,	37.999940272,	-104.000064383,	0x4BAA
18,	12,	9,	-66,	555000000,	2245,	2241,	38.004024385,	-103.990068019,	38.000024500,	-104.000077004,	0xA57D
0,	8,	7,	-101,	560000000,	2253,	2257,	38.003959970,	-103.989905891,	38.000077849,	-103.999940842,	0x3056
6,	13,	14,	-89,	565000000,	2259,	2250,	38.004044338,	-103.989774641,	37.999920654,	-104.000015223,	0x5FE3
12,	9,	12,	-68,	570000000,	2243,	2241,	38.004097827,	-103.989674272,	37.999912244,	-103.999940820,	0x83D3
18,	14,	10,	-61,	575000000,	2242,	2247,	38.004120436,	-103.989604782,	38.000052616,	-104.000076960,	0xAEBD
0,	10,	8,	-102,	580000000,	2257,	2250,	38.004112166,	-103.989566171,	37.999982447,	-104.000064317,	0x7FB8
6,	6,	6,	-87,	585000000,	2250,	2250,	38.004252679,	-103.989558440,	38.000061060,	-104.000082553,	0x0B7F
12,	11,	13,	-82,	590000000,	2259,	2246,	38.004182650,	-103.989401926,	37.999929131,	-103.999952006,	0x5923
18,	7,	11,	-65,	595000000,	2245,	2258,	38.004261404,	-103.989276291,	37.999945985,	-104.000032001,	0x09A7
0,	12,	9,	-97,	600000000,	2247,	2248,	38.004309278,	-103.989181536,	37.999931960,	-103.999963213,	0x250B
6,	8,	7,	-81,	605000000,	2246,	2253,	38.004326273,	-103.989117660,	38.000066718,	-103.999925304,	0xFE6E
12,	13,	14,	-77,	610000000,	2241,	2255,	38.004312388,	-103.989084664,	37.999990934,	-103.999918275,	0x13A6
18,	9,	12,	-65,	615000000,	2252,	2254,	38.004447286,	-103.988902885,	38.000063933,	-103.999942125,	0xE3CE
0,	14,	10,	-102,	620000000,	2241,	2250,	38.004371642,	-103.988931648,	37.999926390,	-103.999996856,	0x7922
6,	10,	8,	-84,	625000000,	2246,	2242,	38.004444782,	-103.988811628,	37.999937629,	-104.000082465,	0xC1B4
12,	6,	6,	-81,	630000000,	2248,	2250,	38.004487042,	-103.988722487,	37.999917990,	-104.000019292,	0x521B
18,	11,	13,	-75,	635000000,	2246,	2255,	38.004498422,	-103.988664226,	38.000047134,	-103.999986997,	0x7FCC
0,	7,	11,	-103,	640000000,	2259,	2257,	38.004658586,	-103.988457181,	37.999965735,	-103.999985583,	0x0960
6,	12,	9,	-82,	645000000,	2251,	2255,	38.004608207,	-103.988460679,	38.000033119,	-104.000015048,	0x2717
12,	8,	7,	-81,	650000000,	2258,	2250,	38.004706612,	-103.988315394,	38.000069624,	-104.000075392,	0xEA12
18,	13,	14,	-79,	655000000,	2244,	2242,	38.004774137,	-103.988200988,	38.000075250,	-103.999986953,	0x6425
0,	9,	12,	-99,	660000000,	2244,	2250,	38.004810782,	-103.988117462,	38.000049996,	-103.999929394,	0x5858
6,	14,	10,	-90,	665000000,	2242,	2254,	38.004816548,	-103.988064815,	37.999993862,	-104.000082378,	0xB4B1
12,	10,	8,	-76,	670000000,	2255,	2255,	38.004791434,	-103.988043048,	38.000086512,	-104.000086577,	0x55C6
18,	6,	6,	-79,	675000000,	2246,	2253,	38.004915104,	-103.987872497,	37.999968619,	-103.999941994,	0x195B
0,	11,	13,	-90,	680000000,	2253,	2248,	38.004828231,	-103.987732826,	37.999999509,	-104.000007953,	0x406C
6,	7,	11,	-93,	685000000,	2256,	2258,	38.004890141,	-103.987624035,	37.999999520,	-103.999925128,	0x40CE
12,	12,	9,	-81,	690000000,	2256,	2246,	38.004921172,	-103.987546123,	37.999968652,	-104.000052847,	0xAFA4
18,	8,	7,	-75,	695000000,	2253,	2250,	38.004921323,	-103.987499091,	38.000086567,	-104.000031782,	0xA57A
0,	13,	14,	-91,	700000000,	2248,	2250,	38.005070258,	-103.987482938,	37.999993939,	-104.000041596,	0x382C
6,	9,	12,	-92,	705000000,	2257,	2247,	38.005008650,	-103.987318002,	38.000050094,	-104.000082290,	0x0AD4
12,	14,	10,	-81,	710000000,	2244,	2241,	38.005095826,	-103.987183946,	38.000075370,	-103.999974200,	0x6ED0
18,	10,	8,	-80,	715000000,	2247,	2250,	38.005152122,	-103.987080769,	38.000069767,	-104.000076653,	0xC283
0,	6,	6,	-101,	720000000,	2247,	2257,	38.005177539,	-103.987008472,	38.000033284,	-104.000030323,	0x03A0
6,	11,	13,	-86,	725000000,	2243,	2241,	38.005172075,	-103.986967054,	37.999965921,	-104.000014872,	0x70B8
12,	7,	11,	-90,	730000000,	2255,	2241,	38.005315396,	-103.986956516,	38.000047342,	-104.000030301,	0x174F
18,	12,	9,	-80,	735000000,	2245,	2256,	38.005248173,	-103.986797194,	37.999918220,	-104.000076610,	0x638D
0,	8,	7,	-92,	740000000,	2250,	2250,	38.005329735,	-103.986668752,	37.999937882,	-103.999974134,	0x2811
6,	13,	14,	-89,	745000000,	2252,	2259,	38.005380416,	-103.986571190,	37.999926664,	-104.000082202,	0x20FE
12,	9,	12,	-95,	750000000,	2251,	2246,	38.005400218,	-103.986504507,	38.000064229,	-104.000041486,	0x851D
18,	14,	10,	-90,	755000000,	2248,	2249,	38.005389141,	-103.986468704,	37.999991252,	-104.000031650,	0xFF59
0,	10,	8,	-93,	760000000,	2259,	2248,	38.005526847,	-103.986284117,	38.000067058,	-104.000052693,	0x194B
6,	6,	6,	-88,	765000000,	2248,	2244,	38.005454010,	-103.986310073,	37.999932322,	-103.999924953,	0xB570
12,	11,	13,	-95,	770000000,	2253,	2255,	38.005529956,	-103.986187245,	37.999946369,	-104.000007756,	0x47CE
18,	7,	11,	-95,	775000000,	2254,	2245,	38.005575024,	-103.986095297,	37.999929537,	-103.999941775,	0x2EEE
0,	12,	9,	-103,	780000000,	2253,	2250,	38.005589211,	-103.986034229,	38.000061488,	-104.000086336,	0x7131
6,	8,	7,	-96,	785000000,	2249,	2251,	38.005572519,	-103.985824377,	37.999982896,	-104.000082114,	0x9A3A
12,	13,	14,	-91,	790000000,	2259,	2250,	38.005704610,	-103.985825068,	38.000053088,	-103.999929109,	0x5942
18,	9,	12,	-96,	795000000,	2248,	2246,	38.005626159,	-103.985676975,	37.999912737,	-103.999986646,	0x518F
0,	14,	10,	-95,	800000000,	2252,	2257,	38.005696491,	-103.985559762,	37.999921170,	-104.000075063,	0x413F
6,	10,	8,	-100,	805000000,	2253,	2246,	38.005735944,	-103.985473429,	38.000078386,	-104.000014697,	0xE7C4
12,	6,	6,	-96,	810000000,	2251,	2250,	38.005744517,	-103.985417975,	38.000025060,	-103.999985210,	0x4136
18,	11,	13,	-91,	815000000,	2246,	2251,	38.005901874,	-103.985393401,	37.999940853,	-103.999986603,	0xFC1F
0,	7,	11,	-96,	820000000,	2256,	2250,	38.005848687,	-103.985220043,	38.000005431,	-104.000018875,	0x98C7
6,	12,	9,	-99,	825000000,	2244,	2245,	38.005944285,	-103.985257228,	38.000039129,	-104.000082027,	0x8CAD
12,	8,	7,	-97,	830000000,	2248,	2255,	38.006009003,	-103.985145629,	38.000041947,	-103.999996395,	0xC18F
18,	13,	14,	-97,	835000000,	2249,	2244,	38.006042841,	-103.985064910,	38.000013885,	-103.999941643,	0x0EBF
0,	9,	12,	-93,	840000000,	2246,	2248,	38.006045799,	-103.984835407,	37.999954944,	-103.999917770,	0x5489
6,	14,	10,	-93,	845000000,	2259,	2249,	38.006017878,	-103.984816448,	38.000044787,	-103.999924778,	0x4B39
12,	10,	8,	-91,	850000000,	2250,	2246,	38.006138741,	-103.984648704,	37.999924087,	-103.999962664,	0x1292
18,	6,	6,	-95,	855000000,	2256,	2259,	38.006228724,	-103.984691504,	37.999952170,	-104.000031431,	0x7A3C
0,	11,	13,	-99,	860000000,	2259,	2250,	38.006108164,	-103.984585520,	37.999949374,	-103.999951413,	0x4B24
6,	7,	11,	-96,	865000000,	2259,	2256,	38.006136387,	-103.984510415,	37.999915698,	-104.000081939,	0x345D
12,	12,	9,	-95,	870000000,	2256,	2241,	38.006313395,	-103.984286527,	38.000030806,	-104.000063681,	0x45CF
18,	8,	7,	-103,	875000000,	2250,	2241,	38.006279859,	-103.984273182,	37.999935371,	-104.000076303,	0x0409
0,	13,	14,	-100,	880000000,	2258,	2257,	38.006395107,	-103.984111053,	37.999988719,	-103.999940141,	0xB8BB
6,	9,	12,	-94,	885000000,	2245,	2250,	38.006299812,	-103.984159467,	38.000011188,	-104.000014521,	0xF6ED
12,	14,	10,	-94,	890000000,	2248,	2241,	38.006353301,	-103.984059097,	38.000002777,	-103.999940119,	0x988A
18,	10,	8,	-92,	895000000,	2247,	2248,	38.006375911,	-103.983989607,	37.999963487,	-104.000076259,	0x48F0
0,	6,	6,	-97,	900000000,	2243,	2250,	38.006547303,	-103.983771333,	38.000072980,	-104.000063615,	0x94DB
6,	11,	13,	-102,	905000000,	2255,	2250,	38.006508153,	-103.983763602,	37.999971930,	-104.000081851,	0xDB82
12,	7,	11,	-102,	910000000,	2245,	2246,	38.006617787,	-103.983607088,	38.000019664,	-103.999951304,	0x757E
18,	12,	9,	-90,	915000000,	2250,	2258,	38.006516878,	-103.983481453,	38.000036519,	-104.000031299,	0x4179
0,	8,	7,	-103,	920000000,	2251,	2248,	38.006564752,	-103.983386698,	38.000022494,	-103.999962511,	0xC248
6,	13,	14,	-91,	925000000,	2250,	2253,	38.006581747,	-103.983322822,	37.999977589,	-103.999924602,	0x83E1
12,	9,	12,	-92,	930000000,	2246,	2255,	38.006747525,	-103.983289826,	38.000081467,	-103.999917573,	0xF800
18,	14,	10,	-98,	935000000,	2257,	2254,	38.006702761,	-103.983287710,	37.999974803,	-103.999941424,	0x9F53
0,	10,	8,	-90,	940000000,	2246,	2250,	38.006806780,	-103.983136810,	38.000016923,	-103.999996154,	0x5321
6,	6,	6,	-90,	945000000,	2250,	2243,	38.006879919,	-103.983016790,	38.000028163,	-104.000081764,	0x1E81
12,	11,	13,	-91,	950000000,	2252,	2250,	38.006742516,	-103.982927649,	38.000008523,	-104.000018590,	0xA0F4
18,	7,	11,	-102,	955000000,	2250,	2255,	38.006933559,	-103.982869388,	37.999958004,	-103.999986296,	0xC099
0,	12,	9,	-101,	960000000,	2245,	2257,	38.006914060,	-103.982842006,	38.000056268,	-103.999984881,	0x8654
6,	8,	7,	-98,	965000000,	2256,	2255,	38.006863681,	-103.982665841,	37.999943990,	-104.000014346,	0x15D7
12,	13,	14,	-100,	970000000,	2244,	2250,	38.006962086,	-103.982520556,	37.999980495,	-104.000074691,	0xF053
18,	9,	12,	-100,	975000000,	2249,	2243,	38.007029611,	-103.982406150,	37.999986120,	-103.999986252,	0x1F5C
0,	14,	10,	-94,	980000000,	2250,	2250,	38.007066256,	-103.982322624,	37.999960866,	-103.999928692,	0xCB13
6,	10,	8,	-102,	985000000,	2247,	2254,	38.007072022,	-103.982269977,	38.000084395,	-104.000081676,	0xAD86
12,	6,	6,	-90,	990000000,	2241,	2255,	38.007046908,	-103.982248210,	37.999997382,	-104.000085876,	0x4EC0
18,	11,	13,	-95,	995000000,	2250,	2253,	38.007170578,	-103.982077660,	38.000059152,	-103.999941292,	0xF17E
0,	7,	11,	-95,	1000000000,	2258,	2248,	38.007263368,	-103.982117652,	37.999910380,	-104.000007251,	0xA2AE
6,	12,	9,	-100,	1005000000,	2242,	2258,	38.007145615,	-103.982008860,	37.999910390,	-103.999924427,	0xD3E9
12,	8,	7,	-103,	1010000000,	2243,	2246,	38.007176646,	-103.981930949,	38.000059185,	-104.000052145,	0x6AAE
18,	13,	14,	-98,	1015000000,	2258,	2250,	38.007356461,	-103.981704253,	37.999997437,	-104.000031080,	0x3F6C
0,	9,	12,	-93,	1020000000,	2252,	2250,	38.007325732,	-103.981688101,	38.000084472,	-104.000040894,	0xBB1B
6,	14,	10,	-95,	1025000000,	2243,	2248,	38.007443788,	-103.981523165,	37.999960965,	-104.000081588,	0x7B21
12,	10,	8,	-98,	1030000000,	2249,	2241,	38.007351300,	-103.981568771,	37.999986241,	-103.999973498,	0xDB28
18,	6,	6,	-97,	1035000000,	2251,	2250,	38.007407596,	-103.981465594,	37.999980637,	-104.000075952,	0x841F
0,	11,	13,	-99,	1040000000,	2251,	2257,	38.007433013,	-103.981393297,	37.999944154,	-104.000029621,	0x0323
6,	7,	11,	-98,	1045000000,	2248,	2241,	38.007427550,	-103.981172216,	38.000056455,	-104.000014171,	0xF4AF
12,	12,	9,	-103,	1050000000,	2241,	2241,	38.007570870,	-103.981161678,	37.999958212,	-104.000029599,	0x7915
18,	8,	7,	-92,	1055000000,	2250,	2257,	38.007503648,	-103.981002356,	38.000008753,	-104.000075908,	0x88F2
0,	13,	14,	-101,	1060000000,	2255,	2250,	38.007585209,	-103.980873914,	38.000028415,	-103.999973433,	0x0DF8
6,	9,	12,	-97,	1065000000,	2257,	2259,	38.007635890,	-103.980776352,	38.000017197,	-104.000081500,	0xE9E4
12,	14,	10,	-102,	1070000000,	2256,	2246,	38.007655692,	-103.980709669,	37.999975099,	-104.000040784,	0xB14C
18,	10,	8,	-95,	1075000000,	2252,	2249,	38.007644615,	-103.980673866,	38.000081785,	-104.000030948,	0xC958
0,	6,	6,	-99,	1080000000,	2245,	2248,	38.007782321,	-103.980668942,	37.999977929,	-104.000051991,	0x996A
6,	11,	13,	-92,	1085000000,	2253,	2244,	38.007709484,	-103.980515235,	38.000022855,	-103.999924251,	0xE223
12,	7,	11,	-97,	1090000000,	2258,	2255,	38.007785431,	-103.980392408,	38.000036903,	-104.000007054,	0x8506
18,	12,	9,	-94,	1095000000,	2241,	2245,	38.007830498,	-103.980300460,	38.000020070,	-103.999941073,	0xAD01
0,	8,	7,	-91,	1100000000,	2258,	2250,	38.007844685,	-103.980239391,	37.999972358,	-104.000085634,	0xC539
6,	13,	14,	-96,	1105000000,	2253,	2252,	38.008007656,	-103.980209202,	38.000073430,	-104.000081413,	0x1D44
12,	9,	12,	-102,	1110000000,	2245,	2250,	38.007960084,	-103.980030230,	37.999963958,	-103.999928407,	0x0E24
18,	14,	10,	-103,	1115000000,	2252,	2246,	38.008061296,	-103.979882137,	38.000003271,	-103.999985945,	0xF8AC
0,	10,	8,	-94,	1120000000,	2257,	2257,	38.008131629,	-103.979944588,	38.000011703,	-104.000074362,	0x3140
6,	6,	6,	-95,	1125000000,	2258,	2246,	38.008171081,	-103.979858254,	37.999989256,	-104.000013995,	0x3C85
12,	11,	13,	-102,	1130000000,	2256,	2250,	38.008179654,	-103.979623137,	37.999935930,	-103.999984508,	0x074D
18,	7,	11,	-93,	1135000000,	2250,	2252,	38.008157348,	-103.979598563,	38.000031387,	-103.999985901,	0x1293
0,	12,	9,	-91,	1140000000,	2242,	2250,	38.008104162,	-103.979604868,	37.999916301,	-104.000018173,	0xDB03
6,	8,	7,	-90,	1145000000,	2249,	2245,	38.008199759,	-103.979462390,	37.999949999,	-104.000081325,	0x5024
12,	13,	14,	-97,	1150000000,	2252,	2255,	38.008264477,	-103.979350791,	37.999952817,	-103.999995693,	0x9484
18,	9,	12,	-93,	1155000000,	2253,	2244,	38.008298315,	-103.979270072,	37.999924756,	-103.999940941,	0x3636
0,	14,	10,	-98,	1160000000,	2250,	2248,	38.008301273,	-103.979220233,	38.000045478,	-103.999917069,	0x6412
6,	10,	8,	-94,	1165000000,	2245,	2249,	38.008453015,	-103.979021610,	37.999955657,	-103.999924076,	0xD0A6
12,	6,	6,	-102,	1170000000,	2254,	2246,	38.008394215,	-103.979033530,	38.000014620,	-103.999961962,	0x2D15
18,	11,	13,	-101,	1175000000,	2242,	2259,	38.008484198,	-103.978896666,	38.000042703,	-104.000030729,	0xF5D1
0,	7,	11,	-101,	1180000000,	2245,	2250,	38.008543301,	-103.978790682,	38.000039907,	-103.999950712,	0x7A82
6,	12,	9,	-94,	1185000000,	2245,	2257,	38.008571525,	-103.978715577,	38.000006231,	-104.000081237,	0xC842
12,	8,	7,	-102,	1190000000,	2242,	2241,	38.008568869,	-103.978671352,	37.999941676,	-104.000062979,	0x55EC
18,	13,	14,	-92,	1195000000,	2254,	2241,	38.008535333,	-103.978478344,	38.000025904,	-104.000075601,	0x03C9
0,	9,	12,	-98,	1200000000,	2244,	2257,	38.008650581,	-103.978495878,	38.000079252,	-103.999939439,	0x792B
6,	14,	10,	-102,	1205000000,	2250,	2250,	38.008734950,	-103.978364629,	37.999922058,	-104.000013820,	0x2106
12,	10,	8,	-97,	1210000000,	2252,	2241,	38.008608775,	-103.978264259,	37.999913647,	-103.999939417,	0x646F
18,	6,	6,	-91,	1215000000,	2251,	2248,	38.008811048,	-103.978194769,	38.000054020,	-104.000075557,	0xE067
0,	11,	13,	-92,	1220000000,	2248,	2250,	38.008802777,	-103.978156159,	37.999983850,	-104.000062913,	0xAFD3
6,	7,	11,	-93,	1225000000,	2241,	2250,	38.008763627,	-103.977968765,	38.000062464,	-104.000081149,	0x8704
12,	12,	9,	-102,	1230000000,	2250,	2246,	38.008873261,	-103.977812250,	37.999930535,	-103.999950602,	0x6E68
18,	8,	7,	-100,	1235000000,	2254,	2258,	38.008952015,	-103.977866279,	37.999947389,	-104.000030597,	0x80D7
0,	13,	14,	-94,	1240000000,	2256,	2248,	38.008999889,	-103.977771523,	37.999933364,	-103.999961809,	0x7E50
6,	9,	12,	-93,	1245000000,	2255,	2253,	38.009016884,	-103.977707648,	38.000068122,	-103.999923900,	0x5A90
"""
debug_text = io.StringIO(debug_text)

if __name__ == '__main__':
    names = [
        'power', 'sats_tx', 'sats_rx', 'rssi',
        'timestamp', 'alt_tx', 'alt_rx',
        'lat_tx', 'lon_tx',
        'lat_rx', 'lon_rx',
        'crc'
    ]

    df = pd.read_csv(debug_text, names=names)
    df['lat_tx'].plot()

    fig, axs = plt.subplots(2, constrained_layout=True)

    axs[0].set_title('Remote Position')
    sns.scatterplot(x='lon_tx', y='lat_tx', data=df, ax=axs[0])
    axs[1].set_title('Base Position')
    sns.scatterplot(x='lon_rx', y='lat_rx', data=df, ax=axs[1])

    fig, ax = plt.subplots(1, constrained_layout=True)
    lat_ref = 38.0
    lon_ref = -104.0
    dist = (40.075e6 / 360) * np.sqrt( (df['lat_tx'] - lat_ref)**2 + (df['lon_tx'] - lon_ref)**2 )
    sns.lineplot(x=dist, y='rssi', hue='power', data=df, ax=ax)

    plt.show()
