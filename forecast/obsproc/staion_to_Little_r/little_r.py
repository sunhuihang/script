import sys

def header_record(Latitude, Longitude, ID, Platform, Elevation, Bogus, Date, SLP,name):

    '''
    Latitude  : F20.5
    Lontitude : F20.5
    Platform  : A40 FM-code
    Elevation : F20.5
    Bogus     : logical
    Date      : A20
    '''
    dict_header = {}
    dict_header['Latitude'            ] = str_F20p5(Latitude)
    dict_header['Longitude'           ] = str_F20p5(Longitude)
    dict_header['ID'                  ] = str_A40_Front_Space(ID)   
    dict_header['Name'                ] = str_A40_Front_Space(name)
    dict_header['Platform'            ] = str_A40_Front_Space(Platform)
    dict_header['Source'              ] = '                                     N/A' # A40   
    dict_header['Elevation'           ] = str_F20p5(Elevation)
    dict_header['ValidFields'         ] = '         1'                               # I10
    dict_header['NumErrors'           ] = '   -888888'                               # I10
    dict_header['NumWarnings'         ] = '   -888888'                               # I10
    dict_header['SequenceNumber'      ] = '       890'                               # I10
    dict_header['NumDuplicates'       ] = '   -888888'                               # I10
    dict_header['IsSounding'          ] = '         F'                               # logical 10
    dict_header['IsBogus'             ] = str_L10(Bogus)
    dict_header['Discard'             ] = '         F'                               # logical 10
    dict_header['UnixTime'            ] = '   -888888'                               # I10
    dict_header['JulianDay'           ] = '   -888888'                               # I10
    dict_header['Date'                ] = str_A20_Behind_Space(Date)
    dict_header['SLP-QC'              ] = str_F13p5(SLP)+'      0'                   # F13.5 I7
    dict_header['RefPressure-QC'      ] = '-888888.00000      0'                     # F13.5 I7
    dict_header['GroundTemp-QC'       ] = '-888888.00000      0'                     # F13.5 I7
    dict_header['SST-QC'              ] = '-888888.00000      0'                     # F13.5 I7
    dict_header['SFCPressure-QC'      ] = '-888888.00000      0'                     # F13.5 I7
    dict_header['Precip-QC'           ] = '-888888.00000      0'                     # F13.5 I7
    dict_header['DailyMaxT-QC'        ] = '-888888.00000      0'                     # F13.5 I7
    dict_header['DailyMinT-QC'        ] = '-888888.00000      0'                     # F13.5 I7
    dict_header['NightMinT-TC'        ] = '-888888.00000      0'                     # F13.5 I7
    dict_header['3hrPresChange-QC'    ] = '-888888.00000      0'                     # F13.5 I7
    dict_header['24hrChange-QC'       ] = '-888888.00000      0'                     # F13.5 I7
    dict_header['CloudCover-QC'       ] = '-888888.00000      0'                     # F13.5 I7
    dict_header['Ceiling-QC'          ] = '-888888.00000      0'                     # F13.5 I7
    dict_header['PrecipitableWater-QC'] = '-888888.00000      0'                     # F13.5 I7
    
    header_str = ''
    for iKey in dict_header.keys():
        header_str = header_str + dict_header[iKey]
    if len(header_str) != 620:
        print("Error: The header record len =",len(header_str),",The correct length should be 620!" )
        sys.exit(1)
    return header_str

def data_record(Pressure, Height, Temperature, DewPoint, WindSpeed, WindDirection, WindU, WindV, RelativeHumidity, Thickness):
    '''
    Pressure         : F13.5, Pa
    Height           : F13.5, m
    Temperature      : F13.5, K
    DewPoint         : F13.5, K
    WindSpeed        : F13.5, m/s
    WindDirection    : F13.5, deg
    WindU            : F13.5, m/s
    WindV            : F13.5, m/s
    RelativeHumidity : F13.5, %
    Thickness        : F13.5, m
    '''
    dict_data = {}
    dict_data['Pressure'        ] = str_F13p5(Pressure) # F13.5
    dict_data['QC1'             ] = '      0'       # I7
    dict_data['Height'          ] = str_F13p5(Height)   # F13.5
    dict_data['QC2'             ] = '      0'       # I7
    dict_data['Temperature'     ] = str_F13p5(Temperature) # F13.5
    dict_data['QC3'             ] = '      0'       # I7
    dict_data['DewPoint'        ] = str_F13p5(DewPoint) # F13.5
    dict_data['QC4'             ] = '      0'       # I7
    dict_data['WindSpeed'       ] = str_F13p5(WindSpeed) # F13.5
    dict_data['QC5'             ] = '      0'       # I7
    dict_data['WindDirection'   ] = str_F13p5(WindDirection) # F13.5
    dict_data['QC6'             ] = '      0'       # I7
    dict_data['WindU'           ] = str_F13p5(WindU) # F13.5
    dict_data['QC7'             ] = '      0'       # I7
    dict_data['WindV'           ] = str_F13p5(WindV) # F13.5
    dict_data['QC8'             ] = '      0'       # I7
    dict_data['RelativeHumidity'] = str_F13p5(RelativeHumidity) # F13.5
    dict_data['QC9'             ] = '      0'       # I7
    dict_data['Thickness'       ] = str_F13p5(Thickness) # F13.5
    dict_data['QC10'            ] = '      0'       # I7

    data_str = ''
    for iKey in dict_data.keys():
        data_str = data_str + dict_data[iKey]
    if len(data_str) != 200:
        print("Error: The data record len =",len(data_str),",The correct length should be 200!" )
        sys.exit(1)
    return data_str

def ending_record():
    ending_str = '-777777.00000      0-777777.00000      0-888888.00000      0-888888.00000      0-888888.00000      0-888888.00000      0-888888.00000      0-888888.00000      0-888888.00000      0-888888.00000      0'
    return ending_str

def tail_record(ValidFields):
    '''
    ValidFields : integer
    '''
    NumErrors   = '      0'
    NumWarnings = '      0'
    tail_str = str_I7(ValidFields) + NumErrors + NumWarnings
    if len(tail_str) != 21:
        print("Error: The tail record len =",len(tail_str),",The correct length should be 21!" )
        sys.exit(1)
    return tail_str

def str_F13p5(float):
    a = ('%.5f' % float)
    space = ' '*(13-len(a))
    return space+str(a)

def str_F20p5(float):
    a = ('%.5f' % float)
    space = ' '*(20-len(a))
    return space+str(a)

def str_A40_Front_Space(string):
    # valid string in Front of space
    string0 = str(string).strip()
    space = ' '*(40-len(string0))
    return string0+space
    
def str_A20_Behind_Space(string):
    # valid string Behind space 
    string0 = str(string).strip()
    space = ' '*(20-len(string0))
    return space+string0

def str_I10(int0):
    space = ' '*(10-len(int0))
    return space+str(int0)

def str_I7(int0):
    space = ' '*(7-len(str(int0)))
    return space+str(int0)

def str_L10(logical):
    if logical:
       string = '         T'
    else:
       string = '         F'
    return string

if __name__ == "__main__":
    # for header
    lat   = 45.0
    lon   = 33.0
    ID    = '76530'
    FM    = 'FM-35 TEMP'
    elev  = 100.0
    bogus = False
    date  = '20210405120000'

    # for data
    pres  = 98000 # Pa
    h     = 101.0 # m
    t     = 237.0 # K
    td    = -888888.00000 # K
    wspd  = 10 # m/s
    wdir  = 210 # deg
    u     = -888888.00000 # m/s
    v     = -888888.00000 # m/s
    rh    = -888888.00000 # %
    tk    = -888888.00000 # m

    # for tail
    num_fields = 1

    header_str = header_record(lat, lon, ID, FM, elev, bogus, date)
    data_str   = data_record(pres, h, t, td, wspd, wdir, u, v, rh, tk)
    ending_str = ending_record()
    tail_str   = tail_record(num_fields)
    print(header_str)
    print(data_str)
    print(ending_str)
    print(tail_str)



