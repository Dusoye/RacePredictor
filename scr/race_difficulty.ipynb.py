### Race difficulty weighting
def aggregate_by_event(df):
    agg_data = df.groupby('Event ID').agg({
        'Distance KM': 'first',
        'Elevation Gain': 'first',
        'Terrain': 'first',
        'Male Finishers': 'first',
        'Female Finishers': 'first',
        'Total Finishers': 'first',
        'Time Seconds Winner': 'first',
        'Time Seconds Finish': 'mean'
    }).reset_index()
    
    # Calculate average time
    agg_data['Average Time Seconds'] = agg_data['Time Seconds Finish']
    
    # Calculate winner and average pace
    agg_data['Winner Pace'] = agg_data['Time Seconds Winner'] / 60 / agg_data['Distance KM']
    agg_data['Average Pace'] = agg_data['Average Time Seconds'] / 60 / agg_data['Distance KM']
    agg_data['Gender Ratio'] = agg_data['Male Finishers'] / agg_data['Total Finishers']
    
    return agg_data

# Apply aggregation
df_event = aggregate_by_event(df_clean)
def sigmoid_normalize(x, midpoint, steepness=1):
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))

def calculate_race_difficulty(df, weights):
    # Normalize factors
    distance_factor = sigmoid_normalize(df['Distance KM'], midpoint=100, steepness=0.05)
    elevation_factor = sigmoid_normalize(df['Elevation Gain'], midpoint=2500, steepness=0.001)
    terrain_factor = {'road': 0.6, 'track': 0.4, 'trail': 1.0, 'other': 0.5}[df['Terrain']]
    finishers_factor = max(1 - (df['Total Finishers'] / 1000), 0.5)
    gender_factor = df['Gender Ratio']
    
    # Calculate difficulty score using provided weights
    difficulty_score = (
        distance_factor * weights['distance'] +
        elevation_factor * weights['elevation'] +
        terrain_factor * weights['terrain'] +
        finishers_factor * weights['finishers'] +
        gender_factor * weights['gender']
    )
    
    return difficulty_score

weights = {
    'distance': 0.3,
    'elevation': 0.3,
    'terrain': 0.2,
    'finishers': 0.1,
    'gender': 0.1
}

df_event['Race Difficulty Score'] = df_event.apply(calculate_race_difficulty, weights=weights, axis=1)
df_event.head()
# Remove events with missing elevation data
agg_df_filtered = df_event.dropna()

print(f"Original number of events: {len(df_event)}")
print(f"Number of events with elevation data: {len(agg_df_filtered)}")
print(f"Number of events removed: {len(df_event) - len(agg_df_filtered)}")
# Initial weights
initial_weights = {
    'distance': -0.3,
    'elevation': -0.3,
    'terrain': -0.2,
    'finishers': -0.1
}

# Calculate initial difficulty scores
agg_df_filtered['Race Difficulty Score'] = agg_df_filtered.apply(lambda row: calculate_race_difficulty(row, initial_weights), axis=1)

def optimize_difficulty_score(df):
    # Prepare features
    X = df[['Distance KM', 'Elevation Gain', 'Total Finishers']]
    X['Terrain'] = df['Terrain'].map({'road': 0.6, 'track': 0.4, 'trail': 1.0, 'other': 0.5})
    
    # Normalize features
    X['Distance KM'] = sigmoid_normalize(X['Distance KM'], midpoint=100, steepness=0.05)
    X['Elevation Gain'] = sigmoid_normalize(X['Elevation Gain'], midpoint=2500, steepness=0.001)
    X['Total Finishers'] = sigmoid_normalize(X['Total Finishers'], midpoint=500, steepness=0.01)
    
    # Use a combination of winner pace and average pace as the target
    y = 1 / (0.7 * df['Winner Pace'] + 0.3 * df['Average Pace'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    # Get optimized weights
    weights = {
        'distance': model.coef_[0],
        'elevation': model.coef_[1],
        'finishers': model.coef_[2],
        'terrain': model.coef_[3]
    }
    print("Optimized weights:", weights)
    
    return weights
# Optimize weights
optimized_weights = optimize_difficulty_score(agg_df_filtered)

# Recalculate difficulty scores with optimized weights
agg_df_filtered['Optimized Difficulty Score'] = agg_df_filtered.apply(lambda row: calculate_race_difficulty(row, optimized_weights), axis=1)

# Compare correlation
baseline_difficulty = 1 / (0.7 * agg_df_filtered['Winner Pace'] + 0.3 * agg_df_filtered['Average Pace'])
initial_correlation = agg_df_filtered['Race Difficulty Score'].corr(baseline_difficulty)
optimized_correlation = agg_df_filtered['Optimized Difficulty Score'].corr(baseline_difficulty)

print(f"Initial correlation: {initial_correlation}")
print(f"Optimized correlation: {optimized_correlation}")
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plotting style
#plt.style.use('seaborn')

# 1. Scatter plot of initial difficulty score vs. baseline difficulty
plt.figure(figsize=(10, 6))
plt.scatter(agg_df_filtered['Race Difficulty Score'], baseline_difficulty, alpha=0.6)
plt.xlabel('Initial Race Difficulty Score')
plt.ylabel('Baseline Difficulty (based on pace)')
plt.title('Initial Difficulty Score vs. Baseline Difficulty')
plt.plot([0, 1], [0, 1], color='r', linestyle='--')  # Add a diagonal line for reference
plt.tight_layout()
plt.show()

# 2. Scatter plot of optimized difficulty score vs. baseline difficulty
plt.figure(figsize=(10, 6))
plt.scatter(agg_df_filtered['Optimized Difficulty Score'], baseline_difficulty, alpha=0.6)
plt.xlabel('Optimized Race Difficulty Score')
plt.ylabel('Baseline Difficulty (based on pace)')
plt.title('Optimized Difficulty Score vs. Baseline Difficulty')
plt.plot([0, 1], [0, 1], color='r', linestyle='--')  # Add a diagonal line for reference
plt.tight_layout()
plt.show()

# 3. Distribution of difficulty scores before and after optimization
plt.figure(figsize=(12, 6))
sns.kdeplot(data=agg_df_filtered['Race Difficulty Score'], shade=True, label='Initial Score')
sns.kdeplot(data=agg_df_filtered['Optimized Difficulty Score'], shade=True, label='Optimized Score')
plt.xlabel('Difficulty Score')
plt.ylabel('Density')
plt.title('Distribution of Difficulty Scores: Initial vs. Optimized')
plt.legend()
plt.tight_layout()
plt.show()

# 4. Bar plot of initial and optimized weights
weights_df = pd.DataFrame({
    'Initial': initial_weights,
    'Optimized': optimized_weights
})

plt.figure(figsize=(10, 6))
weights_df.plot(kind='bar')
plt.title('Comparison of Initial and Optimized Weights')
plt.xlabel('Factors')
plt.ylabel('Weight')
plt.legend(title='Weights')
plt.tight_layout()
plt.show()

# 5. Residual plot for the optimized model
residuals = baseline_difficulty - agg_df_filtered['Optimized Difficulty Score']
plt.figure(figsize=(10, 6))
plt.scatter(agg_df_filtered['Optimized Difficulty Score'], residuals, alpha=0.6)
plt.xlabel('Optimized Difficulty Score')
plt.ylabel('Residuals')
plt.title('Residual Plot for Optimized Difficulty Score')
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.show()

# 6. Difficulty score vs. race distance
plt.figure(figsize=(12, 6))
plt.scatter(agg_df_filtered['Distance KM'], agg_df_filtered['Optimized Difficulty Score'], alpha=0.6)
plt.xlabel('Race Distance (km)')
plt.ylabel('Optimized Difficulty Score')
plt.title('Race Difficulty vs. Distance')
plt.tight_layout()
plt.show()

# Print summary statistics
print("Summary Statistics for Optimized Difficulty Score:")
print(agg_df_filtered['Optimized Difficulty Score'].describe())

# Print correlation matrix
correlation_matrix = agg_df_filtered[['Distance KM', 'Elevation Gain', 'Total Finishers', 'Optimized Difficulty Score']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)
optimized_weights = weights_df['Optimized'].to_dict()
df_event['Optimised Difficulty Score'] = df_event.apply(calculate_race_difficulty, weights=optimized_weights, axis=1)
df_event.head()
if 'Race Difficulty Score' in df_event.columns:
    print("Correlation between original and optimized difficulty scores:")
    print(df_event['Race Difficulty Score'].corr(df_event['Optimised Difficulty Score']))

    # Visualize the difference
    plt.figure(figsize=(10, 6))
    plt.scatter(df_event['Race Difficulty Score'], df_event['Optimised Difficulty Score'], alpha=0.6)
    plt.xlabel('Original Difficulty Score')
    plt.ylabel('Optimised Difficulty Score')
    plt.title('Original vs. Optimised Difficulty Scores')
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')  # Add a diagonal line for reference
    plt.tight_layout()
    plt.show()

# Print summary statistics of the new difficulty scores
print("\nSummary statistics of Optimised Difficulty Scores:")
print(df_event['Optimised Difficulty Score'].describe())

# Visualize the distribution of new difficulty scores
plt.figure(figsize=(10, 6))
sns.histplot(df_event['Optimised Difficulty Score'], kde=True)
plt.title('Distribution of Optimised Difficulty Scores')
plt.xlabel('Difficulty Score')
plt.ylabel('Frequency')
plt.show()
def invert_and_normalize(scores, scale_to_100=False):
    # Invert the scores
    inverted_scores = -scores
    
    # Apply min-max normalization
    normalised = (inverted_scores - inverted_scores.min()) / (inverted_scores.max() - inverted_scores.min())
    
    # Scale to 0-100 range if requested
    if scale_to_100:
        normalised = normalised * 100
    
    return normalised

# Apply the inversion and normalization
df_event['Normalised Difficulty Score'] = invert_and_normalize(df_event['Optimised Difficulty Score'])

# Visualize the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Original Optimised Difficulty Scores
sns.histplot(df_event['Optimised Difficulty Score'], kde=True, ax=ax1)
ax1.set_title('Original Optimised Difficulty Scores')
ax1.set_xlabel('Score')

# Normalised Difficulty Scores
sns.histplot(df_event['Normalised Difficulty Score'], kde=True, ax=ax2)
ax2.set_title('Normalised Difficulty Scores (0-100)')
ax2.set_xlabel('Score')

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary statistics for Original Optimised Difficulty Score:")
print(df_event['Optimised Difficulty Score'].describe())

print("\nSummary statistics for Normalised Difficulty Score:")
print(df_event['Normalised Difficulty Score'].describe())

# Scatter plot to compare original and normalised scores
plt.figure(figsize=(10, 6))
plt.scatter(df_event['Optimised Difficulty Score'], df_event['Normalised Difficulty Score'], alpha=0.6)
plt.xlabel('Original Optimised Difficulty Score')
plt.ylabel('Normalised Difficulty Score')
plt.title('Original vs. Normalised Difficulty Scores')
plt.tight_layout()
plt.show()

# Display the top 10 most difficult races
print("\nTop 10 Most Difficult Races:")
top_10 = df_event.nlargest(10, 'Normalised Difficulty Score')[['Event ID', 'Normalised Difficulty Score', 'Distance KM', 'Elevation Gain']]
print(top_10)

# Display the bottom 10 least difficult races
print("\nBottom 10 Least Difficult Races:")
bottom_10 = df_event.nsmallest(10, 'Normalised Difficulty Score')[['Event ID', 'Normalised Difficulty Score', 'Distance KM', 'Elevation Gain']]
print(bottom_10)
df_clean = df.merge(df_event[['Event ID', 'Normalised Difficulty Score']], on='Event ID', how='left')
df_clean['Inverse Time'] = 1 / (df_clean['Time Seconds Finish'] / 60)
df_clean['Weighted Performance'] = df_clean['Time Seconds Finish'] * df_clean['Normalised Difficulty Score']

df_clean.columns
df_clean.sample(10)