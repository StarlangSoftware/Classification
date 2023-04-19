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
        kMeans.getModel().saveTxt("models/kMeans-iris.txt");
        kMeans.loadModel("models/kMeans-iris.txt");
        assertEquals(7.33, 100 * kMeans.test(iris.getInstanceList()).getErrorRate(), 0.01);
        kMeans.train(bupa.getInstanceList(), kMeansParameter);
        kMeans.getModel().saveTxt("models/kMeans-bupa.txt");
        kMeans.loadModel("models/kMeans-bupa.txt");
        assertEquals(43.77, 100 * kMeans.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        kMeans.train(dermatology.getInstanceList(), kMeansParameter);
        kMeans.getModel().saveTxt("models/kMeans-dermatology.txt");
        kMeans.loadModel("models/kMeans-dermatology.txt");
        assertEquals(45.08, 100 * kMeans.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        kMeans.train(car.getInstanceList(), kMeansParameter);
        kMeans.getModel().saveTxt("models/kMeans-car.txt");
        kMeans.loadModel("models/kMeans-car.txt");
        assertEquals(44.21, 100 * kMeans.test(car.getInstanceList()).getErrorRate(), 0.01);
        kMeans.train(tictactoe.getInstanceList(), kMeansParameter);
        kMeans.getModel().saveTxt("models/kMeans-tictactoe.txt");
        kMeans.loadModel("models/kMeans-tictactoe.txt");
        assertEquals(38.94, 100 * kMeans.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
        kMeans.train(nursery.getInstanceList(), kMeansParameter);
        kMeans.getModel().saveTxt("models/kMeans-nursery.txt");
        kMeans.loadModel("models/kMeans-nursery.txt");
        assertEquals(60.26, 100 * kMeans.test(nursery.getInstanceList()).getErrorRate(), 0.01);
        kMeans.train(chess.getInstanceList(), kMeansParameter);
        kMeans.getModel().saveTxt("models/kMeans-chess.txt");
        kMeans.loadModel("models/kMeans-chess.txt");
        assertEquals(83.25, 100 * kMeans.test(chess.getInstanceList()).getErrorRate(), 0.01);
    }

}