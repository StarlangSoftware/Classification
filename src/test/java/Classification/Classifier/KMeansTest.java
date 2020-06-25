package Classification.Classifier;

import Classification.Parameter.KMeansParameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class KMeansTest extends ClassifierTest{

    @Test
    public void testTrain() {
        KMeans kMeans = new KMeans();
        KMeansParameter kMeansParameter = new KMeansParameter(1);
        kMeans.train(iris.getInstanceList(), kMeansParameter);
        assertEquals(7.33, 100 * kMeans.test(iris.getInstanceList()).getErrorRate(), 0.01);
        kMeans.train(bupa.getInstanceList(), kMeansParameter);
        assertEquals(43.77, 100 * kMeans.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        kMeans.train(dermatology.getInstanceList(), kMeansParameter);
        assertEquals(45.08, 100 * kMeans.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        kMeans.train(car.getInstanceList(), kMeansParameter);
        assertEquals(44.21, 100 * kMeans.test(car.getInstanceList()).getErrorRate(), 0.01);
        kMeans.train(tictactoe.getInstanceList(), kMeansParameter);
        assertEquals(38.94, 100 * kMeans.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
        kMeans.train(nursery.getInstanceList(), kMeansParameter);
        assertEquals(60.26, 100 * kMeans.test(nursery.getInstanceList()).getErrorRate(), 0.01);
        kMeans.train(chess.getInstanceList(), kMeansParameter);
        assertEquals(83.25, 100 * kMeans.test(chess.getInstanceList()).getErrorRate(), 0.01);
    }

}