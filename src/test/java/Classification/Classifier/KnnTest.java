package Classification.Classifier;

import Classification.DistanceMetric.EuclidianDistance;
import Classification.Parameter.KnnParameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class KnnTest extends ClassifierTest{

    @Test
    public void testTrain() {
        Knn knn = new Knn();
        KnnParameter knnParameter = new KnnParameter(1, 3, new EuclidianDistance());
        knn.train(iris.getInstanceList(), knnParameter);
        assertEquals(4.00, 100 * knn.test(iris.getInstanceList()).getErrorRate(), 0.01);
        knn.train(bupa.getInstanceList(), knnParameter);
        assertEquals(19.42, 100 * knn.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        knn.train(dermatology.getInstanceList(), knnParameter);
        assertEquals(3.83, 100 * knn.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        knn.train(car.getInstanceList(), knnParameter);
        assertEquals(21.06, 100 * knn.test(car.getInstanceList()).getErrorRate(), 0.01);
        knn.train(tictactoe.getInstanceList(), knnParameter);
        assertEquals(32.57, 100 * knn.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
        knn.train(nursery.getInstanceList(), knnParameter);
        assertEquals(18.46, 100 * knn.test(nursery.getInstanceList()).getErrorRate(), 0.01);
    }

}