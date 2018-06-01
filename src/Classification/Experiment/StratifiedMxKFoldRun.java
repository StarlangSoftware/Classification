package Classification.Experiment;

import Classification.Classifier.DiscreteFeaturesNotAllowed;
import Classification.Performance.ExperimentPerformance;
import Classification.Instance.Instance;
import Sampling.StratifiedKFoldCrossValidation;

public class StratifiedMxKFoldRun extends MxKFoldRun{

    public StratifiedMxKFoldRun(int M, int K){
        super(M, K);
    }

    public ExperimentPerformance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        ExperimentPerformance result = new ExperimentPerformance();
        for (int j = 0; j < M; j++){
            StratifiedKFoldCrossValidation<Instance> crossValidation = new StratifiedKFoldCrossValidation<>(experiment.getDataSet().getClassInstances(), K, experiment.getParameter().getSeed());
            runExperiment(experiment.getClassifier(), experiment.getParameter(), result, crossValidation);
        }
        return result;
    }

}
