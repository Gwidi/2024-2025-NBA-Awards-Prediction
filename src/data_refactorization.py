import pandas as pd

# Load the CSV files into DataFrames
nba_all_nba_teams = pd.read_csv('../data/nba_all_nba_teams.csv')

# Change the data format from wide to long 
team_levels = ['1st', '2nd', '3rd']
nba_all_nba_teams_long = pd.DataFrame()


for i, team_level in enumerate(team_levels):
    temp_df = nba_all_nba_teams.iloc[i::3].copy() # Select every third row starting from the i-th row
    temp_df = temp_df.melt(id_vars=['Season', 'Tm'], value_vars=['Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'], value_name='Player')
    nba_all_nba_teams_long = pd.concat([nba_all_nba_teams_long, temp_df])

# Add labels
nba_all_nba_teams_long['Label'] = 1

# Delete unnecessary columns
nba_all_nba_teams_long = nba_all_nba_teams_long.drop(columns=['variable'])

nba_all_nba_teams_long.to_csv('../data/nba_all_nba_teams_long.csv', index=False)

# Load the statistics 
league_leaders = pd.read_csv('../data/league_leaders_2000_2024.csv')

# Merge the statistics with the All-NBA teams data
merged_data = pd.merge(league_leaders, nba_all_nba_teams_long[['Season', 'Player', 'Label']], left_on=['PLAYER', 'SEASON'], right_on=['Player', 'Season'], how='left')

# Annotate players not in All-NBA teams with label 0
merged_data['Label'] = merged_data['Label'].fillna(0)

# Drop unnecessary columns
merged_data = merged_data.drop(columns=['Player', 'Season'])

# Save the final dataset
merged_data.to_csv('final_dataset_all_nba_teams.csv', index=False)

