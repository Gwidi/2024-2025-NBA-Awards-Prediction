import pandas as pd
import re

# # Load the CSV files into DataFrames
# nba_all_nba_teams = pd.read_csv('../data/nba_all_nba_teams.csv')

# # Change the data format from wide to long 
# team_levels = ['1st', '2nd', '3rd']
# nba_all_nba_teams_long = pd.DataFrame()


# for i, team_level in enumerate(team_levels):
#     temp_df = nba_all_nba_teams.iloc[i::3].copy() # Select every third row starting from the i-th row
#     temp_df = temp_df.melt(id_vars=['Season', 'Tm'], value_vars=['Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'], value_name='Player')
#     nba_all_nba_teams_long = pd.concat([nba_all_nba_teams_long, temp_df])


# # Delete unnecessary columns
# nba_all_nba_teams_long = nba_all_nba_teams_long.drop(columns=['variable'])

# nba_all_nba_teams_long.to_csv('../data/nba_all_nba_teams_long.csv', index=False)

# # Load the statistics 
# league_leaders = pd.read_csv('../data/league_leaders_2000_2024.csv',index_col=False)

# # Merge the statistics with the All-NBA teams data
# merged_data = pd.merge(league_leaders, nba_all_nba_teams_long[['Season', 'Player', 'Tm']], left_on=['PLAYER', 'SEASON'], right_on=['Player', 'Season'], how='left')

# # Annotate players not in All-NBA teams with label 0
# merged_data['Tm'] = merged_data['Tm'].fillna(0)

# # Drop unnecessary columns
# merged_data = merged_data.drop(columns=['Player', 'Season'])

# # Save the final dataset
# merged_data.to_csv('../data/final_dataset_all_nba_teams.csv', index=False)


# Load All-Rookie Teams data
# Load the existing dataset
league_leaders_df = pd.read_csv("../data/final_dataset_all_nba_teams.csv")
nba_rookie_teams = pd.read_csv('../data/nba_all_nba_rookie.csv')

# Prepare list for (Season, Player Name, Team)
rookie_long = []
for _, row in nba_rookie_teams.iterrows():
    season = row['Season']
    team = row['Tm']  # '1st' or '2nd'
    for player_col in row.index[2:]:
        cell = row[player_col]
        if pd.isna(cell):
            continue
        # Clean cell
        cell_clean = str(cell).replace("(T)", "").strip()
        # Try to extract names (Imię Nazwisko)
        name_list = re.findall(r"[A-Z][a-zA-Z.'-]+ [A-Z][a-zA-Z.'-]+", cell_clean)
        if not name_list:
            name_list = [cell_clean]
        for player in name_list:
            rookie_long.append({
                'Season': season,
                'Player': player.strip(),
                'Team': team
            })
rookie_team_df = pd.DataFrame(rookie_long)

# Normalize player names in your main dataframe and in the rookie teams
def normalize_name(name):
    return name.strip().lower().replace('.', '')

league_leaders_df['PLAYER_NORM'] = league_leaders_df['PLAYER'].apply(normalize_name)
rookie_team_df['PLAYER_NORM'] = rookie_team_df['Player'].apply(normalize_name)

# Create mapping: (Season, Player) -> 1/2
rookie_team_map = {}
for _, row in rookie_team_df.iterrows():
    team_val = 1 if '1' in str(row['Team']) else 2
    rookie_team_map[(row['Season'], row['PLAYER_NORM'])] = team_val

# Add IS_ALL_ROOKIE_TEAM column
def get_rookie_team(row):
    return rookie_team_map.get((row['SEASON'], row['PLAYER_NORM']), 0)

league_leaders_df['IS_ALL_ROOKIE_TEAM'] = league_leaders_df.apply(get_rookie_team, axis=1)

# Optionally drop the helper column
league_leaders_df.drop(columns=['PLAYER_NORM'], inplace=True)

# Save the updated dataframe
league_leaders_df.to_csv("../data/league_leaders_2000_2024_with_rookies_and_all_rookie_team.csv", index=False)

print("IS_ALL_ROOKIE_TEAM column has been added.")

