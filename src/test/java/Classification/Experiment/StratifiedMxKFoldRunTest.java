package Classification.Experiment;

import Classification.Classifier.*;
import Classification.DistanceMetric.EuclidianDistance;
import Classification.Model.*;
import Classification.Model.DecisionTree.DecisionTree;
import Classification.Model.Ensemble.BaggingModel;
import Classification.Model.NeuralNetwork.LinearPerceptronModel;
import Classification.Model.NonParametric.KnnModel;
import Classification.Model.Parametric.LdaModel;
import Classification.Model.Parametric.NaiveBayesModel;
import Classification.Parameter.*;
import Classification.Performance.ExperimentPerformance;
import org.junit.Test;

import static org.junit.Assert.*;

public class StratifiedMxKFoldRunTest extends ClassifierTest {

    @Test
    public void testExecute() throws DiscreteFeaturesNotAllowed {
        StratifiedMxKFoldRun stratifiedMxKFoldRun = new StratifiedMxKFoldRun(5, 2);
        ExperimentPerformance experimentPerformance = stratifiedMxKFoldRun.execute(new Experiment(new DecisionTree(), new C45Parameter(1, true, 0.2), iris));
        assertEquals(4.67, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = stratifiedMxKFoldRun.execute(new Experiment(new DecisionTree(), new C45Parameter(1, true, 0.2), tictactoe));
        assertEquals(19.83, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = stratifiedMxKFoldRun.execute(new Experiment(new KnnModel(), new KnnParameter(1, 3, new EuclidianDistance()), bupa));
        assertEquals(33.62, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = stratifiedMxKFoldRun.execute(new Experiment(new KnnModel(), new KnnParameter(1, 3, new EuclidianDistance()), dermatology));
        assertEquals(15.84, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = stratifiedMxKFoldRun.execute(new Experiment(new LdaModel(), new Parameter(1), bupa));
        assertEquals(31.59, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = stratifiedMxKFoldRun.execute(new Experiment(new LdaModel(), new Parameter(1), dermatology));
        assertEquals(3.54, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = stratifiedMxKFoldRun.execute(new Experiment(new LinearPerceptronModel(), new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), iris));
        assertEquals(2.67, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = stratifiedMxKFoldRun.execute(new Experiment(new LinearPerceptronModel(), new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), dermatology));
        assertEquals(2.73, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = stratifiedMxKFoldRun.execute(new Experiment(new NaiveBayesModel(), new Parameter(1), car));
        assertEquals(15.28, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = stratifiedMxKFoldRun.execute(new Experiment(new NaiveBayesModel(), new Parameter(1), nursery));
        assertEquals(9.81, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = stratifiedMxKFoldRun.execute(new Experiment(new BaggingModel(), new BaggingParameter(1, 50), tictactoe));
        assertEquals(7.20, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = stratifiedMxKFoldRun.execute(new Experiment(new BaggingModel(), new BaggingParameter(1, 50), car));
        assertEquals(7.81, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = stratifiedMxKFoldRun.execute(new Experiment(new DummyModel(), new Parameter(1), nursery));
        assertEquals(66.67, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = stratifiedMxKFoldRun.execute(new Experiment(new DummyModel(), new Parameter(1), iris));
        assertEquals(66.67, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
    }
}