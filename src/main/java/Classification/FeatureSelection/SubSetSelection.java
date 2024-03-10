package Classification.FeatureSelection;

import Classification.Classifier.DiscreteFeaturesNotAllowed;
import Classification.Experiment.Experiment;
import Classification.Experiment.MultipleRun;
import Classification.Performance.ExperimentPerformance;

import java.util.ArrayList;
import java.util.HashSet;

public abstract class SubSetSelection {
    protected FeatureSubSet initialSubSet;

    protected abstract ArrayList<FeatureSubSet> operator(FeatureSubSet current, int numberOfFeatures);

    /**
     * A constructor that sets the initial subset with given input.
     *
     * @param initialSubSet {@link FeatureSubSet} input.
     */
    public SubSetSelection(FeatureSubSet initialSubSet) {
        this.initialSubSet = initialSubSet;
    }

    /**
     * The forward method starts with having no feature in the model. In each iteration, it keeps adding the features that are not currently listed.
     *
     * @param currentSubSetList ArrayList to add the FeatureSubsets.
     * @param current           FeatureSubset that will be added to currentSubSetList.
     * @param numberOfFeatures  The number of features to add the subset.
     */
    protected void forward(ArrayList<FeatureSubSet> currentSubSetList, FeatureSubSet current, int numberOfFeatures) {
        for (int i = 0; i < numberOfFeatures; i++) {
            if (!current.contains(i)) {
                FeatureSubSet candidate = current.clone();
                candidate.add(i);
                currentSubSetList.add(candidate);
            }
        }
    }

    /**
     * The backward method starts with all the features and removes the least significant feature at each iteration.
     *
     * @param currentSubSetList ArrayList to add the FeatureSubsets.
     * @param current           FeatureSubset that will be added to currentSubSetList
     */
    protected void backward(ArrayList<FeatureSubSet> currentSubSetList, FeatureSubSet current) {
        for (int i = 0; i < current.size(); i++) {
            FeatureSubSet candidate = current.clone();
            candidate.remove(i);
            currentSubSetList.add(candidate);
        }
    }

    /**
     * The execute method takes an {@link Experiment} and a {@link MultipleRun} as inputs. By selecting a candidateList from given
     * Experiment it tries to find a FeatureSubSet that gives best performance.
     *
     * @param multipleRun {@link MultipleRun} type input.
     * @param experiment  {@link Experiment} type input.
     * @return FeatureSubSet that gives best performance.
     */
    public FeatureSubSet execute(MultipleRun multipleRun, Experiment experiment) {
        HashSet<FeatureSubSet> processed = new HashSet<>();
        FeatureSubSet best = initialSubSet;
        processed.add(best);
        boolean betterFound = true;
        ExperimentPerformance bestPerformance = null, currentPerformance;
        try {
            if (best.size() > 0){
                bestPerformance = multipleRun.execute(experiment.featureSelectedExperiment(best));
            }
            while (betterFound) {
                betterFound = false;
                ArrayList<FeatureSubSet> candidateList = operator(best, experiment.getDataSet().getDataDefinition().attributeCount());
                for (FeatureSubSet candidateSubSet : candidateList) {
                    if (!processed.contains(candidateSubSet)) {
                        if (candidateSubSet.size() > 0){
                            currentPerformance = multipleRun.execute(experiment.featureSelectedExperiment(candidateSubSet));
                            if (bestPerformance == null || currentPerformance.isBetter(bestPerformance)) {
                                best = candidateSubSet;
                                bestPerformance = currentPerformance;
                                betterFound = true;
                            }
                        }
                        processed.add(candidateSubSet);
                    }
                }
            }
        } catch (DiscreteFeaturesNotAllowed ignored) {
        }
        return best;
    }
}
