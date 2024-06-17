package Classification.Experiment;

import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Performance.ExperimentPerformance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Sampling.KFoldCrossValidation;

import java.util.Random;

public class MxKFoldRunSeparateTest extends KFoldRunSeparateTest {
    protected int M;

    /**
     * Constructor for KFoldRunSeparateTest class. Basically sets K parameter of the K-fold cross-validation and M for
     * the number of times.
     *
     * @param M number of cross-validation times.
     * @param K K of the K-fold cross-validation.
     */
    public MxKFoldRunSeparateTest(int M, int K) {
        super(K);
        this.M = M;
    }

    /**
     * Execute the MxKFold run with separate test set with the given classifier on the given data set using the given parameters.
     *
     * @param experiment Experiment to be run.
     * @return An ExperimentPerformance instance.
     */
    public ExperimentPerformance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        ExperimentPerformance result = new ExperimentPerformance();
        InstanceList instanceList = experiment.getDataSet().getInstanceList();
        Partition partition = new Partition(instanceList, 0.25, new Random(experiment.getParameter().getSeed()), true);
        for (int j = 0; j < M; j++) {
            KFoldCrossValidation<Instance> crossValidation = new KFoldCrossValidation<>(partition.get(1).getInstances(), K, experiment.getParameter().getSeed());
            runExperiment(experiment.getModel(), experiment.getParameter(), result, crossValidation, partition.get(0));
        }
        return result;
    }

}
