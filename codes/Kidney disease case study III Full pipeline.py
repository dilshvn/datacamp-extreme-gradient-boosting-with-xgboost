# Create full pipeline
pipeline = Pipeline([
                     ("featureunion", numeric_categorical_union),
                     ("dictifier", Dictifier()),
                     ("vectorizer", DictVectorizer(sort=False)),
                     ("clf", xgb.XGBClassifier(max_depth=3))
                    ])

# Perform cross-validation
cross_val_scores = cross_val_score(cv=3, estimator=pipeline, X=kidney_data, scoring="roc_auc", y=y)

# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores))