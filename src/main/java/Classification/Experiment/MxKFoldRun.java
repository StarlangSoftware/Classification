package Classification.Experiment;

import Classification.Classifier.DiscreteFeaturesNotAllowed;
import Classification.Performance.ExperimentPerformance;
import Classification.Instance.Instance;
import Sampling.KFoldCrossValidation;

public class MxKFoldRun extends KFoldRun {
    protected int M;

    /**
     * Constructor for MxKFoldRun class. Basically sets K parameter of the K-fold cross-validation and M for the number of times.
     *
     * @param M number of cross-validation times.
     * @param K K of the K-fold cross-validation.
     */
    public MxKFoldRun(int M, int K) {
        super(K);
        this.M = M;
    }

    /**
     * Execute the MxKFold run with the given classifier on the given data set using the given parameters.
     *
     * @param experiment Experiment to be run.
     * @return An array of performances: result. result[i] is the performance of the classifier on the i'th bootstrap run.
     */
    public ExperimentPerformance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        ExperimentPerformance result = new ExperimentPerformance();
        for (int j = 0; j < M; j++) {
            KFoldCrossValidation<Instance> crossValidation = new KFoldCrossValidation<>(experiment.getDataSet().getInstances(), K, experiment.getParameter().getSeed());
            runExperiment(experiment.getClassifier(), experiment.getParameter(), result, crossValidation);
        }
        return result;
    }

}