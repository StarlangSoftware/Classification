package Classification.Experiment;

import Classification.Classifier.*;
import Classification.DistanceMetric.EuclidianDistance;
import Classification.Parameter.*;
import Classification.Performance.ExperimentPerformance;
import org.junit.Test;

import static org.junit.Assert.*;

public class KFoldRunTest extends ClassifierTest {

    @Test
    public void testExecute() throws DiscreteFeaturesNotAllowed {
        KFoldRun kFoldRun = new KFoldRun(10);
        ExperimentPerformance experimentPerformance = kFoldRun.execute(new Experiment(new C45(), new C45Parameter(1, true, 0.2), iris));
        assertEquals(6.00, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = kFoldRun.execute(new Experiment(new C45(), new C45Parameter(1, true, 0.2), tictactoe));
        assertEquals(21.08, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = kFoldRun.execute(new Experiment(new Knn(), new KnnParameter(1, 3, new EuclidianDistance()), bupa));
        assertEquals(35.03, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = kFoldRun.execute(new Experiment(new Knn(), new KnnParameter(1, 3, new EuclidianDistance()), dermatology));
        assertEquals(9.80, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = kFoldRun.execute(new Experiment(new Lda(), new Parameter(1), bupa));
        assertEquals(31.92, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = kFoldRun.execute(new Experiment(new Lda(), new Parameter(1), dermatology));
        assertEquals(2.48, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = kFoldRun.execute(new Experiment(new LinearPerceptron(), new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), iris));
        assertEquals(2.67, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = kFoldRun.execute(new Experiment(new LinearPerceptron(), new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), dermatology));
        assertEquals(4.91, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = kFoldRun.execute(new Experiment(new NaiveBayes(), new Parameter(1), car));
        assertEquals(14.30, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = kFoldRun.execute(new Experiment(new NaiveBayes(), new Parameter(1), nursery));
        assertEquals(9.67, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = kFoldRun.execute(new Experiment(new Bagging(), new BaggingParameter(1, 50), tictactoe));
        assertEquals(3.34, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = kFoldRun.execute(new Experiment(new Bagging(), new BaggingParameter(1, 50), car));
        assertEquals(5.27, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = kFoldRun.execute(new Experiment(new Dummy(), new Parameter(1), nursery));
        assertEquals(66.67, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = kFoldRun.execute(new Experiment(new Dummy(), new Parameter(1), iris));
        assertEquals(78.67, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
    }
}