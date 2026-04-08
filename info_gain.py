# Description: Script for calculating the information gain of each feature by
# running the ASP program with and without the feature and comparing the results.

# Step 1: Fetch the data and create an iteratable feature space

# Step 2: For each feature in the feature space:

    # Step 3: Loop through the data:
        # Step 3.1: Run the ASP program without any features and store the set size
        # Step 3.2: Run the ASP program with the current feature and store the set size
        # Step 3.3: Calculate the information gain as log2(set_size_without_features) - log2(set_size_with_feature)
        # Step 3.4: Add the information gain to a running total for the feature

    # Step 4: Average the information gain for this feature over all the data points and store it in a dictionary

# Step 5: Sort the features by their average information gain and save the results to a file