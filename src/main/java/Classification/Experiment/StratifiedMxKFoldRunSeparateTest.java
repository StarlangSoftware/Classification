package Classification.Experiment;

import Classification.Classifier.DiscreteFeaturesNotAllowed;
import Classification.Performance.ExperimentPerformance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Sampling.StratifiedKFoldCrossValidation;

import java.util.Random;

public class StratifiedMxKFoldRunSeparateTest extends StratifiedKFoldRunSeparateTest {
    protected int M;

    /**
     * Constructor for StratifiedMxKFoldRunSeparateTest class. Basically sets K parameter of the K-fold cross-validation and M for the number of times.
     *
     * @param M number of cross-validation times.
     * @param K K of the K-fold cross-validation.
     */
    public StratifiedMxKFoldRunSeparateTest(int M, int K) {
        super(K);
        this.M = M;
    }

    /**
     * Execute the Stratified MxK-fold cross-validation with the given classifier on the given data set using the given parameters.
     *
     * @param experiment Experiment to be run.
     * @return An array of performances: result. result[i] is the performance of the classifier on the i'th bootstrap run.
     */
    public ExperimentPerformance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        ExperimentPerformance result = new ExperimentPerformance();
        for (int j = 0; j < M; j++) {
            InstanceList instanceList = experiment.getDataSet().getInstanceList();
            Partition partition = instanceList.partition(0.25, new Random(experiment.getParameter().getSeed()));
            StratifiedKFoldCrossValidation<Instance> crossValidation = new StratifiedKFoldCrossValidation<>(partition.get(1).divideIntoClasses().getLists(), K, experiment.getParameter().getSeed());
            runExperiment(experiment.getClassifier(), experiment.getParameter(), result, crossValidation, partition.get(0));
        }
        return result;
    }
}
