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

public class MxKFoldRunTest extends ClassifierTest {

    @Test
    public void testExecute() throws DiscreteFeaturesNotAllowed {
        MxKFoldRun mxKFoldRun = new MxKFoldRun(5, 2);
        ExperimentPerformance experimentPerformance = mxKFoldRun.execute(new Experiment(new DecisionTree(), new C45Parameter(1, true, 0.2), iris));
        assertEquals(6.53, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new DecisionTree(), new C45Parameter(1, true, 0.2), tictactoe));
        assertEquals(22.59, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new KnnModel(), new KnnParameter(1, 3, new EuclidianDistance()), bupa));
        assertEquals(37.39, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new KnnModel(), new KnnParameter(1, 3, new EuclidianDistance()), dermatology));
        assertEquals(16.07, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new LdaModel(), new Parameter(1), bupa));
        assertEquals(32.99, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new LdaModel(), new Parameter(1), dermatology));
        assertEquals(3.72, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new LinearPerceptronModel(), new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), iris));
        assertEquals(6.93, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new LinearPerceptronModel(), new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), dermatology));
        assertEquals(5.08, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new NaiveBayesModel(), new Parameter(1), car));
        assertEquals(15.47, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new NaiveBayesModel(), new Parameter(1), nursery));
        assertEquals(9.72, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new BaggingModel(), new BaggingParameter(1, 50), tictactoe));
        assertEquals(9.37, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new BaggingModel(), new BaggingParameter(1, 50), car));
        assertEquals(9.35, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new DummyModel(), new Parameter(1), nursery));
        assertEquals(66.81, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new DummyModel(), new Parameter(1), iris));
        assertEquals(70.27, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
    }
}