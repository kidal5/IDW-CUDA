#include "ColorPalette.h"

uint32_t ColorPalette::inferno[256] = {
	0x300, 0x400, 0x600, 0x1000700, 0x1010900, 0x1010b00, 0x2010e00,
	0x2021000, 0x3021200, 0x4031400, 0x4031600, 0x5041800, 0x6041b00,
	0x7051d00, 0x8061f00, 0x9062100, 0xa072300, 0xb072600, 0xd082800,
	0xe082a00, 0xf092d00, 0x10092f00, 0x120a3200, 0x130a3400,
	0x140b3600, 0x160b3900, 0x170b3b00, 0x190b3e00, 0x1a0b4000,
	0x1c0c4300, 0x1d0c4500, 0x1f0c4700, 0x200c4a00, 0x220b4c00,
	0x240b4e00, 0x260b5000, 0x270b5200, 0x290b5400, 0x2b0a5600,
	0x2d0a5800, 0x2e0a5a00, 0x300a5c00, 0x32095d00, 0x34095f00,
	0x35096000, 0x37096100, 0x39096200, 0x3b096400, 0x3c096500,
	0x3e096600, 0x40096600, 0x41096700, 0x430a6800, 0x450a6900,
	0x460a6900, 0x480b6a00, 0x4a0b6a00, 0x4b0c6b00, 0x4d0c6b00,
	0x4f0d6c00, 0x500d6c00, 0x520e6c00, 0x530e6d00, 0x550f6d00,
	0x570f6d00, 0x58106d00, 0x5a116d00, 0x5b116e00, 0x5d126e00,
	0x5f126e00, 0x60136e00, 0x62146e00, 0x63146e00, 0x65156e00,
	0x66156e00, 0x68166e00, 0x6a176e00, 0x6b176e00, 0x6d186e00,
	0x6e186e00, 0x70196e00, 0x72196d00, 0x731a6d00, 0x751b6d00,
	0x761b6d00, 0x781c6d00, 0x7a1c6d00, 0x7b1d6c00, 0x7d1d6c00,
	0x7e1e6c00, 0x801f6b00, 0x811f6b00, 0x83206b00, 0x85206a00,
	0x86216a00, 0x88216a00, 0x89226900, 0x8b226900, 0x8d236900,
	0x8e246800, 0x90246800, 0x91256700, 0x93256700, 0x95266600,
	0x96266600, 0x98276500, 0x99286400, 0x9b286400, 0x9c296300,
	0x9e296300, 0xa02a6200, 0xa12b6100, 0xa32b6100, 0xa42c6000,
	0xa62c5f00, 0xa72d5f00, 0xa92e5e00, 0xab2e5d00, 0xac2f5c00,
	0xae305b00, 0xaf315b00, 0xb1315a00, 0xb2325900, 0xb4335800,
	0xb5335700, 0xb7345600, 0xb8355600, 0xba365500, 0xbb375400,
	0xbd375300, 0xbe385200, 0xbf395100, 0xc13a5000, 0xc23b4f00,
	0xc43c4e00, 0xc53d4d00, 0xc73e4c00, 0xc83e4b00, 0xc93f4a00,
	0xcb404900, 0xcc414800, 0xcd424700, 0xcf444600, 0xd0454400,
	0xd1464300, 0xd2474200, 0xd4484100, 0xd5494000, 0xd64a3f00,
	0xd74b3e00, 0xd94d3d00, 0xda4e3b00, 0xdb4f3a00, 0xdc503900,
	0xdd523800, 0xde533700, 0xdf543600, 0xe0563400, 0xe2573300,
	0xe3583200, 0xe45a3100, 0xe55b3000, 0xe65c2e00, 0xe65e2d00,
	0xe75f2c00, 0xe8612b00, 0xe9622a00, 0xea642800, 0xeb652700,
	0xec672600, 0xed682500, 0xed6a2300, 0xee6c2200, 0xef6d2100,
	0xf06f1f00, 0xf0701e00, 0xf1721d00, 0xf2741c00, 0xf2751a00,
	0xf3771900, 0xf3791800, 0xf47a1600, 0xf57c1500, 0xf57e1400,
	0xf6801200, 0xf6811100, 0xf7831000, 0xf7850e00, 0xf8870d00,
	0xf8880c00, 0xf88a0b00, 0xf98c0900, 0xf98e0800, 0xf9900800,
	0xfa910700, 0xfa930600, 0xfa950600, 0xfa970600, 0xfb990600,
	0xfb9b0600, 0xfb9d0600, 0xfb9e0700, 0xfba00700, 0xfba20800,
	0xfba40a00, 0xfba60b00, 0xfba80d00, 0xfbaa0e00, 0xfbac1000,
	0xfbae1200, 0xfbb01400, 0xfbb11600, 0xfbb31800, 0xfbb51a00,
	0xfbb71c00, 0xfbb91e00, 0xfabb2100, 0xfabd2300, 0xfabf2500,
	0xfac12800, 0xf9c32a00, 0xf9c52c00, 0xf9c72f00, 0xf8c93100,
	0xf8cb3400, 0xf8cd3700, 0xf7cf3a00, 0xf7d13c00, 0xf6d33f00,
	0xf6d54200, 0xf5d74500, 0xf5d94800, 0xf4db4b00, 0xf4dc4f00,
	0xf3de5200, 0xf3e05600, 0xf3e25900, 0xf2e45d00, 0xf2e66000,
	0xf1e86400, 0xf1e96800, 0xf1eb6c00, 0xf1ed7000, 0xf1ee7400,
	0xf1f07900, 0xf1f27d00, 0xf2f38100, 0xf2f48500, 0xf3f68900,
	0xf4f78d00, 0xf5f89100, 0xf6fa9500, 0xf7fb9900, 0xf9fc9d00,
	0xfafda000, 0xfcfea400
};