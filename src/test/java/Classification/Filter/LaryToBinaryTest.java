package Classification.Filter;

import Classification.Classifier.C45;
import Classification.Classifier.ClassifierTest;
import Classification.Classifier.Knn;
import Classification.DistanceMetric.EuclidianDistance;
import Classification.Parameter.C45Parameter;
import Classification.Parameter.KnnParameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class LaryToBinaryTest extends ClassifierTest {

    @Test
    public void testKnn() {
        Knn knn = new Knn();
        KnnParameter knnParameter = new KnnParameter(1, 3, new EuclidianDistance());
        LaryToBinary laryToBinary = new LaryToBinary(car);
        laryToBinary.convert();
        knn.train(car.getInstanceList(), knnParameter);
        assertEquals(21.06, 100 * knn.test(car.getInstanceList()).getErrorRate(), 0.01);
        laryToBinary = new LaryToBinary(tictactoe);
        laryToBinary.convert();
        knn.train(tictactoe.getInstanceList(), knnParameter);
        assertEquals(32.57, 100 * knn.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
    }

    @Test
    public void testC45() {
        C45 c45 = new C45();
        C45Parameter c45Parameter = new C45Parameter(1, true, 0.2);
        LaryToBinary laryToBinary = new LaryToBinary(car);
        laryToBinary.convert();
        c45.train(car.getInstanceList(), c45Parameter);
        assertEquals(2.78, 100 * c45.test(car.getInstanceList()).getErrorRate(), 0.01);
        laryToBinary = new LaryToBinary(tictactoe);
        laryToBinary.convert();
        c45.train(tictactoe.getInstanceList(), c45Parameter);
        assertEquals(4.49, 100 * c45.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
        laryToBinary = new LaryToBinary(nursery);
        laryToBinary.convert();
        c45.train(nursery.getInstanceList(), c45Parameter);
        assertEquals(0.52, 100 * c45.test(nursery.getInstanceList()).getErrorRate(), 0.01);
    }

}