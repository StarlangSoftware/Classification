package Classification.Experiment;

import Classification.Classifier.DiscreteFeaturesNotAllowed;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Performance.Performance;
import Sampling.StratifiedKFoldCrossValidation;

public class StratifiedSingleRunWithK {

    private int K;

    public StratifiedSingleRunWithK(int K){
        this.K = K;
    }

    public Performance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        StratifiedKFoldCrossValidation<Instance> crossValidation = new StratifiedKFoldCrossValidation<>(experiment.getDataSet().getClassInstances(), K, experiment.getParameter().getSeed());
        InstanceList trainSet = new InstanceList(crossValidation.getTrainFold(0));
        InstanceList testSet = new InstanceList(crossValidation.getTestFold(0));
        return experiment.getClassifier().singleRun(experiment.getParameter(), trainSet, testSet);
    }
}
