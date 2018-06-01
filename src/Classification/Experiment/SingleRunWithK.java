package Classification.Experiment;

import Classification.Classifier.Classifier;
import Classification.Classifier.DiscreteFeaturesNotAllowed;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Parameter.Parameter;
import Classification.Performance.Performance;
import Sampling.CrossValidation;
import Sampling.KFoldCrossValidation;

public class SingleRunWithK implements SingleRun {
    private int K;

    public SingleRunWithK(int K){
        this.K = K;
    }

    protected Performance runExperiment(Classifier classifier, Parameter parameter, CrossValidation<Instance> crossValidation) throws DiscreteFeaturesNotAllowed{
        InstanceList trainSet = new InstanceList(crossValidation.getTrainFold(0));
        InstanceList testSet = new InstanceList(crossValidation.getTestFold(0));
        return classifier.singleRun(parameter, trainSet, testSet);
    }

    public Performance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        KFoldCrossValidation<Instance> crossValidation = new KFoldCrossValidation<>(experiment.getDataSet().getInstances(), K, experiment.getParameter().getSeed());
        return runExperiment(experiment.getClassifier(), experiment.getParameter(), crossValidation);
    }
}
