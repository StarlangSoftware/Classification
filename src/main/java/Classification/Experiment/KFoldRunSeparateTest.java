package Classification.Experiment;

import Classification.Classifier.Classifier;
import Classification.Classifier.DiscreteFeaturesNotAllowed;
import Classification.Performance.ExperimentPerformance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Classification.Parameter.Parameter;
import Sampling.CrossValidation;
import Sampling.KFoldCrossValidation;

import java.util.Random;

public class KFoldRunSeparateTest extends KFoldRun{
    /**
     * Constructor for KFoldRunSeparateTest class. Basically sets K parameter of the K-fold cross-validation.
     *
     * @param K K of the K-fold cross-validation.
     */
    public KFoldRunSeparateTest(int K) {
        super(K);
    }

    protected void runExperiment(Classifier classifier, Parameter parameter, ExperimentPerformance experimentPerformance, CrossValidation<Instance> crossValidation, InstanceList testSet) throws DiscreteFeaturesNotAllowed {
        for (int i = 0; i < K; i++){
            InstanceList trainSet = new InstanceList(crossValidation.getTrainFold(i));
            classifier.train(trainSet, parameter);
            experimentPerformance.add(classifier.test(testSet));
        }
    }

    /**
     * Execute K-fold cross-validation with separate test set with the given classifier on the given data set using the given parameters.
     * @param experiment Experiment to be run.
     * @return An array of performances: result. result[i] is the performance of the classifier on the i'th fold.
     */
    public ExperimentPerformance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        ExperimentPerformance result = new ExperimentPerformance();
        InstanceList instanceList = experiment.getDataSet().getInstanceList();
        Partition partition = instanceList.partition(0.25, new Random(experiment.getParameter().getSeed()));
        KFoldCrossValidation<Instance> crossValidation = new KFoldCrossValidation<>(partition.get(1).getInstances(), K, experiment.getParameter().getSeed());
        runExperiment(experiment.getClassifier(), experiment.getParameter(), result, crossValidation, partition.get(0));
        return result;
    }

}
