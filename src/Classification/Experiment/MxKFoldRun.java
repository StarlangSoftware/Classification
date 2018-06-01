package Classification.Experiment;

import Classification.Classifier.DiscreteFeaturesNotAllowed;
import Classification.Performance.ExperimentPerformance;
import Classification.Instance.Instance;
import Sampling.KFoldCrossValidation;

public class MxKFoldRun extends KFoldRun{
    protected int M;

    public MxKFoldRun(int M, int K){
        super(K);
        this.M = M;
    }

    public ExperimentPerformance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        ExperimentPerformance result = new ExperimentPerformance();
        for (int j = 0; j < M; j++){
            KFoldCrossValidation<Instance> crossValidation = new KFoldCrossValidation<>(experiment.getDataSet().getInstances(), K, experiment.getParameter().getSeed());
            runExperiment(experiment.getClassifier(), experiment.getParameter(), result, crossValidation);
        }
        return result;
    }

}
