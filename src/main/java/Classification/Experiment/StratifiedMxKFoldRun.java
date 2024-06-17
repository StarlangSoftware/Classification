package Classification.Experiment;

import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Performance.ExperimentPerformance;
import Classification.Instance.Instance;
import Sampling.StratifiedKFoldCrossValidation;

public class StratifiedMxKFoldRun extends MxKFoldRun {

    /**
     * Constructor for StratifiedMxKFoldRun class. Basically sets K parameter of the K-fold cross-validation and M for the number of times.
     *
     * @param M number of cross-validation times.
     * @param K K of the K-fold cross-validation.
     */
    public StratifiedMxKFoldRun(int M, int K) {
        super(M, K);
    }

    /**
     * Execute the Stratified MxK-fold cross-validation with the given classifier on the given data set using the given parameters.
     *
     * @param experiment Experiment to be run.
     * @return An ExperimentPerformance instance.
     */
    public ExperimentPerformance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        ExperimentPerformance result = new ExperimentPerformance();
        for (int j = 0; j < M; j++) {
            StratifiedKFoldCrossValidation<Instance> crossValidation = new StratifiedKFoldCrossValidation<>(experiment.getDataSet().getClassInstances(), K, experiment.getParameter().getSeed());
            runExperiment(experiment.getModel(), experiment.getParameter(), result, crossValidation);
        }
        return result;
    }

}