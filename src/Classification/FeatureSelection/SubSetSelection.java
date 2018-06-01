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

    public SubSetSelection(FeatureSubSet initialSubSet){
        this.initialSubSet = initialSubSet;
    }

    protected void forward(ArrayList<FeatureSubSet> currentSubSetList, FeatureSubSet current, int numberOfFeatures){
        for (int i = 0; i < numberOfFeatures; i++){
            if (!current.contains(i)){
                FeatureSubSet candidate = current.clone();
                candidate.add(i);
                currentSubSetList.add(candidate);
            }
        }
    }

    protected void backward(ArrayList<FeatureSubSet> currentSubSetList, FeatureSubSet current){
        for (int i = 0; i < current.size(); i++){
            FeatureSubSet candidate = current.clone();
            candidate.remove(i);
            currentSubSetList.add(candidate);
        }
    }

    public FeatureSubSet execute(MultipleRun multipleRun, Experiment experiment){
        HashSet<FeatureSubSet> processed = new HashSet<>();
        FeatureSubSet best = initialSubSet;
        processed.add(best);
        boolean betterFound = true;
        ExperimentPerformance bestPerformance, currentPerformance;
        try {
            bestPerformance = multipleRun.execute(experiment.featureSelectedExperiment(best));
            while (betterFound){
                betterFound = false;
                ArrayList<FeatureSubSet> candidateList = operator(best, experiment.getDataSet().getDataDefinition().attributeCount());
                for (FeatureSubSet candidateSubSet : candidateList){
                    if (!processed.contains(candidateSubSet)){
                        currentPerformance = multipleRun.execute(experiment.featureSelectedExperiment(candidateSubSet));
                        if (currentPerformance.isBetter(bestPerformance)){
                            best = candidateSubSet;
                            bestPerformance = currentPerformance;
                            betterFound = true;
                        }
                        processed.add(candidateSubSet);
                    }
                }
            }
        } catch (DiscreteFeaturesNotAllowed discreteFeaturesNotAllowed) {
            discreteFeaturesNotAllowed.printStackTrace();
        }
        return best;
    }
}
