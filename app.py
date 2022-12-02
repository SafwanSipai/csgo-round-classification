from numpy import int64
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Default values and session states
###############################################################################################################

maps = ['de_cache', 'de_dust2', 'de_inferno', 'de_mirage', 'de_nuke', 'de_overpass', 'de_train', 'de_vertigo']
primary_weapon = ['-', 'Ak47', 'Aug', 'Awp', 'Bizon', 'Famas', 'G3sg1', 'Galilar', 'M249', 
'M4a1s', 'M4a4', 'Mac10', 'Mag7', 'Mp5sd', 'Mp7', 'Mp9', 'Negev', 'Nova', 'P90', 'R8revolver', 'Sawedoff', 
'Scar20', 'Sg553', 'Ssg08', 'Ump45', 'Xm1014']
t_secondary_weapon = ['Glock', 'Elite', 'Cz75auto', 'Deagle', 'Fiveseven', 'Usps', 'P250', 'P2000', 'Tec9']
ct_secondary_weapon = ['Usps', 'Elite', 'Cz75auto', 'Deagle', 'Fiveseven', 'Glock', 'P250', 'P2000', 'Tec9']

df = pd.read_csv('input.csv')
columns = df.columns

mean_std = {'time_left': (97.88692206521264, 54.465237956138175), 'ct_health': (412.10656809084225, 132.293290207345), 't_health': (402.7145004493097, 139.9190329110423), 'ct_armor': (314.1421207417695, 171.02973554649972), 't_armor': (298.44466955314107, 174.5765451676254), 'ct_money': (9789.023772567602, 11215.042285717855), 't_money': (11241.036680009804, 12162.806759335292), 'ct_helmets': (2.053900825095989, 1.8414700082426372), 't_helmets': (2.7736377746916103, 2.010914530151846), 'ct_defuse_kits': (1.6137243689241074, 1.605780347746835), 'ct_players_alive': (4.2737521444326445, 1.2055003522795404), 't_players_alive': (4.266187402989952, 1.228325054574696)}

###############################################################################################################

st.title('CSGO Round Winner Predictor')
st.write('''
A web application to decide the outcome of a competitive round of CSGO by taking different parameters
into account.''')
st.markdown('---')

model = st.radio('Choose the model', horizontal=True,
options=['Logistic Regression', 'SVM', 'Random Forest Classifier', 'Neural Network'])
map = st.selectbox('Select a map', maps)
time_left = st.slider('Time left', 0, 180)
ct_score = st.slider('CT score', 0, 15)
t_score = st.slider('T score', 0, 15)
bomb_planted = int(st.checkbox('Bomb planted'))

df.at[0, 'bomb_planted'] = int(bomb_planted)
df.at[0,'time_left'] = int(time_left)
df.at[0,'ct_score'] = int(ct_score)
df.at[0,'t_score'] = int(t_score)
df.at[0,'map_de_cache'] = 1 if map == 'de_cache' else 0
df.at[0,'map_de_dust2'] = 1 if map == 'de_dust2' else 0
df.at[0,'map_de_inferno'] = 1 if map == 'de_inferno' else 0
df.at[0,'map_de_mirage'] = 1 if map == 'de_mirage' else 0
df.at[0,'map_de_nuke'] = 1 if map == 'de_nuke' else 0
df.at[0,'map_de_overpass'] = 1 if map == 'de_overpass' else 0
df.at[0,'map_de_train'] = 1 if map == 'de_train' else 0
df.at[0,'map_de_vertigo'] = 1 if map == 'de_vertigo' else 0

st.markdown('---')


ct, t = st.columns(2)

st.markdown('---')

with ct:
    ct_primary = dict.fromkeys(primary_weapon, 0)
    ct_secondary = dict.fromkeys(ct_secondary_weapon, 0)
    
    st.header('CT')
    ct_players_alive = st.slider('Players alive', 1, 5, key='ct_alive')
    ct_money = st.slider('Team money ($)', 0, ct_players_alive*16000, step=50, key='ct_money')
    ct_health = st.slider('Players\' total health', 0, 100*ct_players_alive, key='ct_health')
    ct_armor = st.slider('Players\' total armor', 0, 100*ct_players_alive, key='ct_armor')
    ct_helmets = st.slider('Total helmeted players', 0, ct_players_alive, key='ct_helmets')
    ct_defuse_kits = st.slider('Total defuse kits', 0, ct_players_alive, key='ct_defuse')
    ct_hegrenade = st.slider('HE grenades', 0, ct_players_alive, key='ct_he')
    ct_smokegrenade = st.slider('Smoke grenades', 0, ct_players_alive, key='ct_smoke')
    ct_flashbang = st.slider('Flashbangs', 0, ct_players_alive*2, key='ct_flash')
    ct_incendiarygrenade = st.slider('Incendiary grenade', 0, ct_players_alive, key='ct_incendiary')
    ct_molotovgrenade = st.slider('Molotovs', 0, ct_players_alive, key='ct_molotov')
    ct_decoygrenade = st.slider('Decoy grenade', 0, ct_players_alive, key='ct_decoy')

    st.markdown('---')

    for i in range(ct_players_alive):

        st.subheader('Player ' + str(i+1))
        p_weapon = st.selectbox('Primary Weapon', primary_weapon, key='ct_primary_'+str(i+1))
        s_weapon = st.selectbox('Secondary Weapon', ct_secondary_weapon, key='ct_secondary_'+str(i+1))
        ct_primary[p_weapon] += 1
        ct_secondary[s_weapon] += 1

    df.at[0, 'ct_players_alive'] = int(ct_players_alive)
    df.at[0, 'ct_money'] = int(ct_money)
    df.at[0, 'ct_health'] = int(ct_health)
    df.at[0,'ct_armor'] = int(ct_armor)
    df.at[0,'ct_helmets'] = int(ct_helmets)
    df.at[0,'ct_defuse_kits'] = int(ct_defuse_kits)
    df.at[0,'ct_grenade_hegrenade'] = int(ct_hegrenade)
    df.at[0,'ct_grenade_smokegrenade'] = int(ct_smokegrenade)
    df.at[0,'ct_grenade_flashbang'] = int(ct_flashbang)
    df.at[0,'ct_grenade_incendiarygrenade'] = int(ct_incendiarygrenade)
    df.at[0,'ct_grenade_molotovgrenade'] = int(ct_molotovgrenade)
    df.at[0,'ct_grenade_decoygrenade'] = int(ct_decoygrenade)


    df.at[0, 'ct_weapon_ak47'] = int(ct_primary['Ak47'])
    df.at[0, 'ct_weapon_aug'] = int(ct_primary['Aug'])
    df.at[0, 'ct_weapon_awp'] = int(ct_primary['Awp'])
    df.at[0, 'ct_weapon_bizon'] = int(ct_primary['Bizon'])
    df.at[0, 'ct_weapon_famas'] = int(ct_primary['Famas'])
    df.at[0, 'ct_weapon_g3sg1'] = int(ct_primary['G3sg1'])
    df.at[0, 'ct_weapon_galilar'] = int(ct_primary['Galilar'])
    df.at[0, 'ct_weapon_m249'] = int(ct_primary['M249'])
    df.at[0, 'ct_weapon_m4a1s'] = int(ct_primary['M4a1s'])
    df.at[0, 'ct_weapon_m4a4'] = int(ct_primary['M4a4'])
    df.at[0, 'ct_weapon_mac10'] = int(ct_primary['Mac10'])
    df.at[0, 'ct_weapon_mag7'] = int(ct_primary['Mag7'])
    df.at[0, 'ct_weapon_mp5sd'] = int(ct_primary['Mp5sd'])
    df.at[0, 'ct_weapon_mp7'] = int(ct_primary['Mp7'])
    df.at[0, 'ct_weapon_mp9'] = int(ct_primary['Mp9'])
    df.at[0, 'ct_weapon_negev'] = int(ct_primary['Negev'])
    df.at[0, 'ct_weapon_nova'] = int(ct_primary['Nova'])
    df.at[0, 'ct_weapon_p90'] = int(ct_primary['P90'])
    df.at[0, 'ct_weapon_r8revolver'] = int(ct_primary['R8revolver'])
    df.at[0, 'ct_weapon_sawedoff'] = int(ct_primary['Sawedoff'])
    df.at[0, 'ct_weapon_scar20'] = int(ct_primary['Scar20'])
    df.at[0, 'ct_weapon_sg553'] = int(ct_primary['Sg553'])
    df.at[0, 'ct_weapon_ssg08'] = int(ct_primary['Ssg08'])
    df.at[0, 'ct_weapon_ump45'] = int(ct_primary['Ump45'])
    df.at[0, 'ct_weapon_xm1014'] = int(ct_primary['Xm1014'])

    df.at[0, 'ct_weapon_cz75auto'] = int(ct_secondary['Cz75auto'])
    df.at[0, 'ct_weapon_elite'] = int(ct_secondary['Elite'])
    df.at[0, 'ct_weapon_glock'] = int(ct_secondary['Glock'])
    df.at[0, 'ct_weapon_deagle'] = int(ct_secondary['Deagle'])
    df.at[0, 'ct_weapon_fiveseven'] = int(ct_secondary['Fiveseven'])
    df.at[0, 'ct_weapon_usps'] = int(ct_secondary['Usps'])
    df.at[0, 'ct_weapon_p250'] = int(ct_secondary['P250'])
    df.at[0, 'ct_weapon_p2000'] = int(ct_secondary['P2000'])
    df.at[0, 'ct_weapon_tec9'] = int(ct_secondary['Tec9'])

with t:
    t_primary = dict.fromkeys(primary_weapon, 0)
    t_secondary = dict.fromkeys(t_secondary_weapon, 0)

    st.header('T')
    t_players_alive = st.slider('Players alive', 1, 5, key='t_alive')
    t_money = st.slider('Team money ($)', 0, t_players_alive*16000, step=50, key='t_money')
    t_health = st.slider('Players\' total health', 0, 100*t_players_alive, key='t_health')
    t_armor = st.slider('Players\' total armor', 0, 100*t_players_alive, key='t_armor')
    t_helmets = st.slider('Total helmeted players', 0, t_players_alive, key='t_helmets')
    t_defuse_kits = st.slider('Total defuse kits', 0, t_players_alive, key='t_defuse', disabled=True)
    t_defuse_kits = 0
    t_hegrenade = st.slider('HE grenades', 0, t_players_alive, key='t_he')
    t_smokegrenade = st.slider('Smoke grenades', 0, t_players_alive, key='t_smoke')
    t_flashbang = st.slider('Flashbangs', 0, t_players_alive*2, key='t_flash')
    t_incendiarygrenade = st.slider('Incendiary grenade', 0, t_players_alive, key='t_incendiary')
    t_molotovgrenade = st.slider('Molotovs', 0, t_players_alive, key='t_molotov')
    t_decoygrenade = st.slider('Decoy grenade', 0, t_players_alive, key='t_decoy')


    st.markdown('---')

    for i in range(t_players_alive):

        st.subheader('Player ' + str(i+1))
        p_weapon = st.selectbox('Primary Weapon', primary_weapon, key='t_primary_'+str(i+1))
        s_weapon = st.selectbox('Secondary Weapon', t_secondary_weapon, key='t_secondary_'+str(i+1))
        t_primary[p_weapon] += 1
        t_secondary[s_weapon] += 1

    df.at[0, 't_players_alive'] = int(t_players_alive)
    df.at[0, 't_money'] = int(t_money)
    df.at[0, 't_health'] = int(t_health)
    df.at[0, 't_armor'] = int(t_armor)
    df.at[0, 't_helmets'] = int(t_helmets)
    df.at[0, 't_grenade_hegrenade'] = int(t_hegrenade)
    df.at[0, 't_grenade_smokegrenade'] = int(t_smokegrenade)
    df.at[0, 't_grenade_flashbang'] = int(t_flashbang)
    df.at[0, 't_grenade_incendiarygrenade'] = int(t_incendiarygrenade)
    df.at[0, 't_grenade_molotovgrenade'] = int(t_molotovgrenade)
    df.at[0, 't_grenade_decoygrenade'] = int(t_decoygrenade)


    df.at[0, 't_weapon_ak47'] = int(t_primary['Ak47'])
    df.at[0, 't_weapon_aug'] = int(t_primary['Aug'])
    df.at[0, 't_weapon_awp'] = int(t_primary['Awp'])
    df.at[0, 't_weapon_bizon'] = int(t_primary['Bizon'])
    df.at[0, 't_weapon_famas'] = int(t_primary['Famas'])
    df.at[0, 't_weapon_g3sg1'] = int(t_primary['G3sg1'])
    df.at[0, 't_weapon_galilar'] = int(t_primary['Galilar'])
    df.at[0, 't_weapon_m249'] = int(t_primary['M249'])
    df.at[0, 't_weapon_m4a1s'] = int(t_primary['M4a1s'])
    df.at[0, 't_weapon_m4a4'] = int(t_primary['M4a4'])
    df.at[0, 't_weapon_mac10'] = int(t_primary['Mac10'])
    df.at[0, 't_weapon_mag7'] = int(t_primary['Mag7'])
    df.at[0, 't_weapon_mp5sd'] = int(t_primary['Mp5sd'])
    df.at[0, 't_weapon_mp7'] = int(t_primary['Mp7'])
    df.at[0, 't_weapon_mp9'] = int(t_primary['Mp9'])
    df.at[0, 't_weapon_negev'] = int(t_primary['Negev'])
    df.at[0, 't_weapon_nova'] = int(t_primary['Nova'])
    df.at[0, 't_weapon_p90'] = int(t_primary['P90'])
    df.at[0, 't_weapon_r8revolver'] = int(t_primary['R8revolver'])
    df.at[0, 't_weapon_sawedoff'] = int(t_primary['Sawedoff'])
    df.at[0, 't_weapon_scar20'] = int(t_primary['Scar20'])
    df.at[0, 't_weapon_sg553'] = int(t_primary['Sg553'])
    df.at[0, 't_weapon_ssg08'] = int(t_primary['Ssg08'])
    df.at[0, 't_weapon_ump45'] = int(t_primary['Ump45'])
    df.at[0, 't_weapon_xm1014'] = int(t_primary['Xm1014'])

    df.at[0, 't_weapon_cz75auto'] = int(t_secondary['Cz75auto'])
    df.at[0, 't_weapon_elite'] = int(t_secondary['Elite'])
    df.at[0, 't_weapon_glock'] = int(t_secondary['Glock'])
    df.at[0, 't_weapon_deagle'] = int(t_secondary['Deagle'])
    df.at[0, 't_weapon_fiveseven'] = int(t_secondary['Fiveseven'])
    df.at[0, 't_weapon_usps'] = int(t_secondary['Usps'])
    df.at[0, 't_weapon_p250'] = int(t_secondary['P250'])
    df.at[0, 't_weapon_p2000'] = int(t_secondary['P2000'])
    df.at[0, 't_weapon_tec9'] = int(t_secondary['Tec9'])


if model == 'Logistic Regression':
        model = pickle.load(open('logistic.sav', 'rb'))
elif model == 'SVM':
    model = pickle.load(open('svm.sav', 'rb'))
elif model == 'Random Forest Classifier':
    model = pickle.load(open('randomForest.sav', 'rb'))
elif model == 'Neural Network':
    model = pickle.load(open('nnet.h5', 'rb'))

normalize_columns = ['time_left', 'ct_health', 't_health', 'ct_armor', 't_armor', 'ct_money', 't_money', 
    'ct_helmets', 't_helmets', 'ct_defuse_kits', 'ct_players_alive', 't_players_alive']

for col in normalize_columns:
    df[col] = (df[col]-mean_std[col][0])/mean_std[col][1]

X = df.drop(columns=['round_winner'], inplace=False, axis=1)

prediction = model.predict(X)
if prediction[0][0] == 'C':
    result = 'Counter Terrorists'
else:
    result = 'Terrorists'

st.header('Winner Prediction: ' + result)