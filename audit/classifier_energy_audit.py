# first thing i will need to import all the tools i will use it in the writing script

#imports the needed library 
import pandas as pd
import time
import os
from codecarbon import EmissionsTracker


#import the classifiers i will use
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


# and then the last import is to import the datasets and preprocessing 
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# and then the second part is to define the classidiers and the datasets after import them into the code
classifiers = {
    "Nearest Neighbors": KNeighborsClassifier(3),
    "Linear SVM":        SVC(kernel="linear", C=0.025, random_state=42),
    "RBF SVM":           SVC(gamma=2, C=1, random_state=42),
    "Gaussian Process":  GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    "Decision Tree":     DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":     RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
    "Neural Network":    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    "AdaBoost":          AdaBoostClassifier(random_state=42),
    "Naive Bayes":       GaussianNB(),
    "QDA":               QuadraticDiscriminantAnalysis(),
}

datasets = {
    "Linear":  make_classification(n_features=2, n_redundant=0,
                                    n_informative=2, random_state=1,
                                    n_clusters_per_class=1),
    "Moons":   make_moons(noise=0.3, random_state=0),
    "Circles": make_circles(noise=0.2, factor=0.5, random_state=1),
}

# now until now we finished the first 2 sections that we write it as summary in the log 

#now starting the third section which is the measurement loop 
#first will define the results to store on it 
results=[]

#and will make file for it to be saved their in the end
os.makedirs("audit/results", exist_ok=True)

# now for each combination of the 3 datasets and the 10 classifiers we will measure
# now making loop for iterating over each dataset
for dataset_name, (X, y) in datasets.items():

    # here we just making normalization cause there is models like svm works better when data is scaled
    X = StandardScaler().fit_transform(X)
    # then splitting the data into 60:40
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
# here we make the second loop for iterate over each classifier and that to make all the cominations
    for clf_name, clf in classifiers.items():

        # but first we need to define the codecarbon tracker cause i will start it before training and then stop it after training
        # and the goal of the tracker here is to measure the energy used and the co2
        tracker = EmissionsTracker(
            project_name=f"{clf_name}_{dataset_name}",
            output_dir="audit/results",
            log_level="error",
            save_to_file=False,
        )

#starting measure and starting calculating the time
        tracker.start()
        start_time = time.time()

# then start training the models
        clf.fit(X_train, y_train)        

# see how long it takes 
        duration  = time.time() - start_time
        #and stop the tracker
        emissions = tracker.stop()
#and we will also calculating the accuracy cause we want to balance between the accuracy and the used energy 
        score = clf.score(X_test, y_test)

        results.append({
            "classifier":   clf_name,
            "dataset":      dataset_name,
            "accuracy":     round(score, 4),
            "duration_sec": round(duration, 4),
            "co2_kg":       emissions,
        })

        print(f"{clf_name:25s} | {dataset_name:8s} | "
              f"acc={score:.2f} | t={duration:.3f}s | CO2={emissions:.2e} kg")
        

# after that we need to save and summarize the results
df = pd.DataFrame(results)
df.to_csv("audit/results/emissions.csv", index=False)

print("\n Mean CO₂ per classifier (across all datasets) ")
summary = (df.groupby("classifier")[["co2_kg","duration_sec","accuracy"]]
             .mean()
             .sort_values("co2_kg", ascending=False))
print(summary.to_string())
print("\nResults saved to audit/results/emissions.csv")