package Classification.Experiment;

import Classification.Classifier.Classifier;
import Classification.Classifier.DiscreteFeaturesNotAllowed;
import Classification.Performance.ExperimentPerformance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Parameter.Parameter;
import Sampling.CrossValidation;
import Sampling.KFoldCrossValidation;

public class KFoldRun implements MultipleRun {
    protected int K;

    /**
     * Constructor for KFoldRun class. Basically sets K parameter of the K-fold cross-validation.
     *
     * @param K K of the K-fold cross-validation.
     */
    public KFoldRun(int K) {
        this.K = K;
    }

    protected void runExperiment(Classifier classifier, Parameter parameter, ExperimentPerformance experimentPerformance, CrossValidation<Instance> crossValidation) throws DiscreteFeaturesNotAllowed {
        for (int i = 0; i < K; i++) {
            InstanceList trainSet = new InstanceList(crossValidation.getTrainFold(i));
            InstanceList testSet = new InstanceList(crossValidation.getTestFold(i));
            classifier.train(trainSet, parameter);
            experimentPerformance.add(classifier.test(testSet));
        }
    }

    /**
     * Execute K-fold cross-validation with the given classifier on the given data set using the given parameters.
     *
     * @param experiment Experiment to be run.
     * @return An array of performances: result. result[i] is the performance of the classifier on the i'th fold.
     */
    public ExperimentPerformance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        ExperimentPerformance result = new ExperimentPerformance();
        KFoldCrossValidation<Instance> crossValidation = new KFoldCrossValidation<>(experiment.getDataSet().getInstances(), K, experiment.getParameter().getSeed());
        runExperiment(experiment.getClassifier(), experiment.getParameter(), result, crossValidation);
        return result;
    }
}
