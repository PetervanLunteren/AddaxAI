"""
AddaxAI Geographic Data

Country and US state data with flag emojis and ISO codes for location selection.

Note: Workaround for Claude Code API quirk - it fails on plain US state names.
To avoid this, we prefix states with emojis, split them in two parts, and map them to their codes.
"""

# Workaround for Claude Code API quirk: it fails on plain US state names.
# To avoid this, we prefix states with emojis, split them in two parts, and map them to their codes.

# Country data with flag emojis and ISO country codes
countries_data = {
    "∅ Do not limit predictions to one country": "NONE",
    "🇦🇼 Aruba": "ABW",
    "🇦🇫 Afghanistan": "AFG",
    "🇦🇴 Angola": "AGO",
    "🇦🇮 Anguilla": "AIA",
    "🇦🇽 Åland Islands": "ALA",
    "🇦🇱 Albania": "ALB",
    "🇦🇩 Andorra": "AND",
    "🇦🇪 United Arab Emirates": "ARE",
    "🇦🇷 Argentina": "ARG",
    "🇦🇲 Armenia": "ARM",
    "🇦🇸 American Samoa": "ASM",
    "🇦🇶 Antarctica": "ATA",
    "🇹🇫 French Southern Territories": "ATF",
    "🇦🇬 Antigua and Barbuda": "ATG",
    "🇦🇺 Australia": "AUS",
    "🇦🇹 Austria": "AUT",
    "🇦🇿 Azerbaijan": "AZE",
    "🇧🇮 Burundi": "BDI",
    "🇧🇪 Belgium": "BEL",
    "🇧🇯 Benin": "BEN",
    "🇧🇶 Bonaire, Sint Eustatius and Saba": "BES",
    "🇧🇫 Burkina Faso": "BFA",
    "🇧🇩 Bangladesh": "BGD",
    "🇧🇬 Bulgaria": "BGR",
    "🇧🇭 Bahrain": "BHR",
    "🇧🇸 Bahamas": "BHS",
    "🇧🇦 Bosnia and Herzegovina": "BIH",
    "🇧🇱 Saint Barthélemy": "BLM",
    "🇧🇾 Belarus": "BLR",
    "🇧🇿 Belize": "BLZ",
    "🇧🇲 Bermuda": "BMU",
    "🇧🇴 Bolivia": "BOL",
    "🇧🇷 Brazil": "BRA",
    "🇧🇧 Barbados": "BRB",
    "🇧🇳 Brunei": "BRN",
    "🇧🇹 Bhutan": "BTN",
    "🇧🇻 Bouvet Island": "BVT",
    "🇧🇼 Botswana": "BWA",
    "🇨🇫 Central African Republic": "CAF",
    "🇨🇦 Canada": "CAN",
    "🇨🇨 Cocos Islands": "CCK",
    "🇨🇭 Switzerland": "CHE",
    "🇨🇱 Chile": "CHL",
    "🇨🇳 China": "CHN",
    "🇨🇮 Côte d'Ivoire": "CIV",
    "🇨🇲 Cameroon": "CMR",
    "🇨🇩 Congo (Democratic Republic)": "COD",
    "🇨🇬 Congo": "COG",
    "🇨🇰 Cook Islands": "COK",
    "🇨🇴 Colombia": "COL",
    "🇰🇲 Comoros": "COM",
    "🇨🇻 Cabo Verde": "CPV",
    "🇨🇷 Costa Rica": "CRI",
    "🇨🇺 Cuba": "CUB",
    "🇨🇼 Curaçao": "CUW",
    "🇨🇽 Christmas Island": "CXR",
    "🇰🇾 Cayman Islands": "CYM",
    "🇨🇾 Cyprus": "CYP",
    "🇨🇿 Czechia": "CZE",
    "🇩🇪 Germany": "DEU",
    "🇩🇯 Djibouti": "DJI",
    "🇩🇲 Dominica": "DMA",
    "🇩🇰 Denmark": "DNK",
    "🇩🇴 Dominican Republic": "DOM",
    "🇩🇿 Algeria": "DZA",
    "🇪🇨 Ecuador": "ECU",
    "🇪🇬 Egypt": "EGY",
    "🇪🇷 Eritrea": "ERI",
    "🇪🇭 Western Sahara": "ESH",
    "🇪🇸 Spain": "ESP",
    "🇪🇪 Estonia": "EST",
    "🇪🇹 Ethiopia": "ETH",
    "🇫🇮 Finland": "FIN",
    "🇫🇯 Fiji": "FJI",
    "🇫🇰 Falkland Islands": "FLK",
    "🇫🇷 France": "FRA",
    "🇫🇴 Faroe Islands": "FRO",
    "🇫🇲 Micronesia": "FSM",
    "🇬🇦 Gabon": "GAB",
    "🇬🇧 United Kingdom": "GBR",
    "🇬🇪 Georgia": "GEO",
    "🇬🇬 Guernsey": "GGY",
    "🇬🇭 Ghana": "GHA",
    "🇬🇮 Gibraltar": "GIB",
    "🇬🇳 Guinea": "GIN",
    "🇬🇵 Guadeloupe": "GLP",
    "🇬🇲 Gambia": "GMB",
    "🇬🇼 Guinea-Bissau": "GNB",
    "🇬🇶 Equatorial Guinea": "GNQ",
    "🇬🇷 Greece": "GRC",
    "🇬🇩 Grenada": "GRD",
    "🇬🇱 Greenland": "GRL",
    "🇬🇹 Guatemala": "GTM",
    "🇬🇫 French Guiana": "GUF",
    "🇬🇺 Guam": "GUM",
    "🇬🇾 Guyana": "GUY",
    "🇭🇰 Hong Kong": "HKG",
    "🇭🇲 Heard Island and McDonald Islands": "HMD",
    "🇭🇳 Honduras": "HND",
    "🇭🇷 Croatia": "HRV",
    "🇭🇹 Haiti": "HTI",
    "🇭🇺 Hungary": "HUN",
    "🇮🇩 Indonesia": "IDN",
    "🇮🇲 Isle of Man": "IMN",
    "🇮🇳 India": "IND",
    "🇮🇴 British Indian Ocean Territory": "IOT",
    "🇮🇪 Ireland": "IRL",
    "🇮🇷 Iran": "IRN",
    "🇮🇶 Iraq": "IRQ",
    "🇮🇸 Iceland": "ISL",
    "🇮🇱 Israel": "ISR",
    "🇮🇹 Italy": "ITA",
    "🇯🇲 Jamaica": "JAM",
    "🇯🇪 Jersey": "JEY",
    "🇯🇴 Jordan": "JOR",
    "🇯🇵 Japan": "JPN",
    "🇰🇿 Kazakhstan": "KAZ",
    "🇰🇪 Kenya": "KEN",
    "🇰🇬 Kyrgyzstan": "KGZ",
    "🇰🇭 Cambodia": "KHM",
    "🇰🇮 Kiribati": "KIR",
    "🇰🇳 Saint Kitts and Nevis": "KNA",
    "🇰🇷 South Korea": "KOR",
    "🇰🇼 Kuwait": "KWT",
    "🇱🇦 Laos": "LAO",
    "🇱🇧 Lebanon": "LBN",
    "🇱🇷 Liberia": "LBR",
    "🇱🇾 Libya": "LBY",
    "🇱🇨 Saint Lucia": "LCA",
    "🇱🇮 Liechtenstein": "LIE",
    "🇱🇰 Sri Lanka": "LKA",
    "🇱🇸 Lesotho": "LSO",
    "🇱🇹 Lithuania": "LTU",
    "🇱🇺 Luxembourg": "LUX",
    "🇱🇻 Latvia": "LVA",
    "🇲🇴 Macao": "MAC",
    "🇲🇫 Saint Martin": "MAF",
    "🇲🇦 Morocco": "MAR",
    "🇲🇨 Monaco": "MCO",
    "🇲🇩 Moldova": "MDA",
    "🇲🇬 Madagascar": "MDG",
    "🇲🇻 Maldives": "MDV",
    "🇲🇽 Mexico": "MEX",
    "🇲🇭 Marshall Islands": "MHL",
    "🇲🇰 North Macedonia": "MKD",
    "🇲🇱 Mali": "MLI",
    "🇲🇹 Malta": "MLT",
    "🇲🇲 Myanmar": "MMR",
    "🇲🇪 Montenegro": "MNE",
    "🇲🇳 Mongolia": "MNG",
    "🇲🇵 Northern Mariana Islands": "MNP",
    "🇲🇿 Mozambique": "MOZ",
    "🇲🇷 Mauritania": "MRT",
    "🇲🇸 Montserrat": "MSR",
    "🇲🇺 Mauritius": "MUS",
    "🇲🇼 Malawi": "MWI",
    "🇲🇾 Malaysia": "MYS",
    "🇾🇹 Mayotte": "MYT",
    "🇳🇦 Namibia": "NAM",
    "🇳🇨 New Caledonia": "NCL",
    "🇳🇪 Niger": "NER",
    "🇳🇫 Norfolk Island": "NFK",
    "🇳🇬 Nigeria": "NGA",
    "🇳🇮 Nicaragua": "NIC",
    "🇳🇺 Niue": "NIU",
    "🇳🇱 Netherlands": "NLD",
    "🇳🇴 Norway": "NOR",
    "🇳🇵 Nepal": "NPL",
    "🇳🇷 Nauru": "NRU",
    "🇳🇿 New Zealand": "NZL",
    "🇴🇲 Oman": "OMN",
    "🇵🇰 Pakistan": "PAK",
    "🇵🇦 Panama": "PAN",
    "🇵🇳 Pitcairn": "PCN",
    "🇵🇪 Peru": "PER",
    "🇵🇭 Philippines": "PHL",
    "🇵🇼 Palau": "PLW",
    "🇵🇬 Papua New Guinea": "PNG",
    "🇵🇱 Poland": "POL",
    "🇵🇷 Puerto Rico": "PRI",
    "🇰🇵 North Korea": "PRK",
    "🇵🇹 Portugal": "PRT",
    "🇵🇾 Paraguay": "PRY",
    "🇵🇸 Palestine": "PSE",
    "🇵🇫 French Polynesia": "PYF",
    "🇶🇦 Qatar": "QAT",
    "🇷🇪 Réunion": "REU",
    "🇷🇴 Romania": "ROU",
    "🇷🇺 Russia": "RUS",
    "🇷🇼 Rwanda": "RWA",
    "🇸🇦 Saudi Arabia": "SAU",
    "🇸🇩 Sudan": "SDN",
    "🇸🇳 Senegal": "SEN",
    "🇸🇬 Singapore": "SGP",
    "🇬🇸 South Georgia": "SGS",
    "🇸🇭 Saint Helena": "SHN",
    "🇸🇯 Svalbard and Jan Mayen": "SJM",
    "🇸🇧 Solomon Islands": "SLB",
    "🇸🇱 Sierra Leone": "SLE",
    "🇸🇻 El Salvador": "SLV",
    "🇸🇲 San Marino": "SMR",
    "🇸🇴 Somalia": "SOM",
    "🇵🇲 Saint Pierre and Miquelon": "SPM",
    "🇷🇸 Serbia": "SRB",
    "🇸🇸 South Sudan": "SSD",
    "🇸🇹 Sao Tome and Principe": "STP",
    "🇸🇷 Suriname": "SUR",
    "🇸🇰 Slovakia": "SVK",
    "🇸🇮 Slovenia": "SVN",
    "🇸🇪 Sweden": "SWE",
    "🇸🇿 Eswatini": "SWZ",
    "🇸🇽 Sint Maarten": "SXM",
    "🇸🇨 Seychelles": "SYC",
    "🇸🇾 Syria": "SYR",
    "🇹🇨 Turks and Caicos Islands": "TCA",
    "🇹🇩 Chad": "TCD",
    "🇹🇬 Togo": "TGO",
    "🇹🇭 Thailand": "THA",
    "🇹🇯 Tajikistan": "TJK",
    "🇹🇰 Tokelau": "TKL",
    "🇹🇱 Timor-Leste": "TLS",
    "🇹🇲 Turkmenistan": "TKM",
    "🇹🇳 Tunisia": "TUN",
    "🇹🇴 Tonga": "TON",
    "🇹🇷 Turkey": "TUR",
    "🇹🇹 Trinidad and Tobago": "TTO",
    "🇹🇻 Tuvalu": "TUV",
    "🇹🇼 Taiwan": "TWN",
    "🇹🇿 Tanzania": "TZA",
    "🇺🇬 Uganda": "UGA",
    "🇺🇦 Ukraine": "UKR",
    "🇺🇲 United States Minor Outlying Islands": "UMI",
    "🇺🇾 Uruguay": "URY",
    "🇺🇸 United States": "USA",
    "🇺🇿 Uzbekistan": "UZB",
    "🇻🇦 Vatican City": "VAT",
    "🇻🇨 Saint Vincent and the Grenadines": "VCT",
    "🇻🇪 Venezuela": "VEN",
    "🇻🇬 British Virgin Islands": "VGB",
    "🇻🇮 US Virgin Islands": "VIR",
    "🇻🇳 Vietnam": "VNM",
    "🇻🇺 Vanuatu": "VUT",
    "🇼🇫 Wallis and Futuna": "WLF",
    "🇼🇸 Samoa": "WSM",
    "🇽🇰 Kosovo": "XKX",
    "🇾🇪 Yemen": "YEM",
    "🇿🇦 South Africa": "ZAF",
    "🇿🇲 Zambia": "ZMB",
    "🇿🇼 Zimbabwe": "ZWE"
}



# US states with two-letter abbreviations
# Due to some weird Claude Code API behaviour, it can’t handle plain US state names.
# Hence, we prefix each state with an emoji and extract codes from here.

us_states_part_one = {
    "🐘 Alabama": "AL",             # Crimson Tide mascot
    "🐻 Alaska": "AK",              # Brown bears & wilderness
    "🌵 Arizona": "AZ",             # Desert cactus
    "💎 Arkansas": "AR",            # Crater of Diamonds State Park
    "🌴 California": "CA",          # Palm trees & beaches
    "⛰️ Colorado": "CO",            # Rocky Mountains
    "⚓ Connecticut": "CT",          # Maritime history
    "🏰 Delaware": "DE",            # Colonial history, "First State"
    "🐊 Florida": "FL",             # Alligators
    "🍑 Georgia": "GA",             # Peaches
    "🌺 Hawaii": "HI",              # Hibiscus, state flower
    "🥔 Idaho": "ID",               # Famous for potatoes
    "🏙️ Illinois": "IL",            # Chicago skyline
    "🏎️ Indiana": "IN",             # Indy 500
    "🌽 Iowa": "IA",                # Corn production
    "🦅 Kansas": "KS",              # Plains wildlife & state bird
    "🥃 Kentucky": "KY",            # Bourbon whiskey
    "🎷 Louisiana": "LA",           # Jazz, New Orleans
    "🦞 Maine": "ME"                # Lobsters
}

us_states_part_two = {
    "🦀 Maryland": "MD",            # Blue crabs
    "🍏 Massachusetts": "MA",       # Apples, colonial heritage
    "🚗 Michigan": "MI",            # Motor City (Detroit)
    "🦌 Minnesota": "MN",           # Deer & lake wildlife
    "🎸 Mississippi": "MS",         # Birthplace of blues
    "🎵 Missouri": "MO",            # St. Louis & Kansas City music
    "🐎 Montana": "MT",             # Horses & open plains
    "🌾 Nebraska": "NE",            # Wheat & prairies
    "🎰 Nevada": "NV",              # Las Vegas casinos
    "🍁 New Hampshire": "NH",       # Fall foliage
    "🎢 New Jersey": "NJ",          # Boardwalk amusement parks
    "🌞 New Mexico": "NM",          # Desert sun & Zia symbol
    "🗽 New York": "NY",            # Statue of Liberty
    "🏖️ North Carolina": "NC",      # Outer Banks beaches
    "🐂 North Dakota": "ND",        # Bison & plains
    "🎭 Ohio": "OH",                # Rock & Roll Hall of Fame, arts
    "🌪️ Oklahoma": "OK",            # Tornado Alley
    "🌲 Oregon": "OR",              # Evergreen forests
    "🔔 Pennsylvania": "PA",        # Liberty Bell
    "🦪 Rhode Island": "RI",        # Oysters & seafood
    "🌊 South Carolina": "SC",      # Atlantic beaches
    "🪨 South Dakota": "SD",        # Badlands & Mount Rushmore
    "🎤 Tennessee": "TN",           # Country music, Nashville
    "🤠 Texas": "TX",               # Cowboy culture
    "🏜️ Utah": "UT",                # Red rock canyons
    "🥞 Vermont": "VT",             # Maple syrup
    "⚔️ Virginia": "VA",            # Colonial & Revolutionary history
    "☕ Washington": "WA",           # Coffee culture (Seattle)
    "⛏️ West Virginia": "WV",       # Coal mining
    "🧀 Wisconsin": "WI",           # Cheese production
    "🦬 Wyoming": "WY",             # Yellowstone bison
    "🏛️ District of Columbia": "DC" # US Capitol
}
us_states_data = {
    "∅ Do not limit predictions to one state": "NONE",
    **us_states_part_one,
    **us_states_part_two
}
