package Classification.Experiment;

import Classification.Classifier.DiscreteFeaturesNotAllowed;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Performance.Performance;
import Sampling.StratifiedKFoldCrossValidation;

public class StratifiedSingleRunWithK {

    private int K;


    /**
     * Constructor for StratifiedSingleRunWithK class. Basically sets K parameter of the K-fold cross-validation.
     *
     * @param K K of the K-fold cross-validation.
     */
    public StratifiedSingleRunWithK(int K) {
        this.K = K;
    }

    /**
     * Execute Stratified Single K-fold cross-validation with the given classifier on the given data set using the given parameters.
     *
     * @param experiment Experiment to be run.
     * @return An array of performances: result. result[i] is the performance of the classifier on the i'th fold.
     * @throws DiscreteFeaturesNotAllowed Exception for discrete features.
     */
    public Performance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        StratifiedKFoldCrossValidation<Instance> crossValidation = new StratifiedKFoldCrossValidation<>(experiment.getDataSet().getClassInstances(), K, experiment.getParameter().getSeed());
        InstanceList trainSet = new InstanceList(crossValidation.getTrainFold(0));
        InstanceList testSet = new InstanceList(crossValidation.getTestFold(0));
        return experiment.getClassifier().singleRun(experiment.getParameter(), trainSet, testSet);
    }
}
