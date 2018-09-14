package Classification.Experiment;

import Classification.Classifier.DiscreteFeaturesNotAllowed;
import Classification.Performance.ExperimentPerformance;
import Classification.Instance.Instance;
import Sampling.StratifiedKFoldCrossValidation;

public class StratifiedKFoldRun extends KFoldRun {

    /**
     * Constructor for KFoldRun class. Basically sets K parameter of the K-fold cross-validation.
     *
     * @param K K of the K-fold cross-validation.
     */
    public StratifiedKFoldRun(int K) {
        super(K);
    }

    /**
     * Execute Stratified K-fold cross-validation with the given classifier on the given data set using the given parameters.
     *
     * @param experiment Experiment to be run.
     * @return An array of performances: result. result[i] is the performance of the classifier on the i'th fold.
     */
    public ExperimentPerformance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        ExperimentPerformance result = new ExperimentPerformance();
        StratifiedKFoldCrossValidation<Instance> crossValidation = new StratifiedKFoldCrossValidation<>(experiment.getDataSet().getClassInstances(), K, experiment.getParameter().getSeed());
        runExperiment(experiment.getClassifier(), experiment.getParameter(), result, crossValidation);
        return result;
    }
}
