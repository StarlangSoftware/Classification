package Classification.Experiment;

import Classification.Classifier.*;
import Classification.DistanceMetric.EuclidianDistance;
import Classification.Parameter.*;
import Classification.Performance.ExperimentPerformance;
import org.junit.Test;

import static org.junit.Assert.*;

public class MxKFoldRunTest extends ClassifierTest {

    @Test
    public void testExecute() throws DiscreteFeaturesNotAllowed {
        MxKFoldRun mxKFoldRun = new MxKFoldRun(5, 2);
        ExperimentPerformance experimentPerformance = mxKFoldRun.execute(new Experiment(new C45(), new C45Parameter(1, true, 0.2), iris));
        assertEquals(6.53, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new C45(), new C45Parameter(1, true, 0.2), tictactoe));
        assertEquals(22.59, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new Knn(), new KnnParameter(1, 3, new EuclidianDistance()), bupa));
        assertEquals(37.39, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new Knn(), new KnnParameter(1, 3, new EuclidianDistance()), dermatology));
        assertEquals(16.07, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new Lda(), new Parameter(1), bupa));
        assertEquals(32.99, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new Lda(), new Parameter(1), dermatology));
        assertEquals(3.72, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new LinearPerceptron(), new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), iris));
        assertEquals(7.73, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new LinearPerceptron(), new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), dermatology));
        assertEquals(4.86, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new NaiveBayes(), new Parameter(1), car));
        assertEquals(15.47, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new NaiveBayes(), new Parameter(1), nursery));
        assertEquals(9.72, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new Bagging(), new BaggingParameter(1, 50), tictactoe));
        assertEquals(9.37, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new Bagging(), new BaggingParameter(1, 50), car));
        assertEquals(9.35, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new Dummy(), new Parameter(1), nursery));
        assertEquals(66.81, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = mxKFoldRun.execute(new Experiment(new Dummy(), new Parameter(1), iris));
        assertEquals(70.27, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
    }
}