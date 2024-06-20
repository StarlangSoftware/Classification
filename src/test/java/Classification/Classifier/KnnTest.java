package Classification.Classifier;

import Classification.DistanceMetric.EuclidianDistance;
import Classification.Model.NonParametric.KnnModel;
import Classification.Parameter.KnnParameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class KnnTest extends ClassifierTest{

    @Test
    public void testTrain() {
        KnnModel knn = new KnnModel();
        KnnParameter knnParameter = new KnnParameter(1, 3, new EuclidianDistance());
        knn.train(iris.getInstanceList(), knnParameter);
        knn.saveTxt("models/knn-iris.txt");
        knn.loadModel("models/knn-iris.txt");
        assertEquals(4.00, 100 * knn.test(iris.getInstanceList()).getErrorRate(), 0.01);
        knn.train(bupa.getInstanceList(), knnParameter);
        knn.saveTxt("models/knn-bupa.txt");
        knn.loadModel("models/knn-bupa.txt");
        assertEquals(19.42, 100 * knn.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        knn.train(dermatology.getInstanceList(), knnParameter);
        knn.saveTxt("models/knn-dermatology.txt");
        knn.loadModel("models/knn-dermatology.txt");
        assertEquals(3.83, 100 * knn.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        knn.train(car.getInstanceList(), knnParameter);
        knn.saveTxt("models/knn-car.txt");
        knn.loadModel("models/knn-car.txt");
        assertEquals(21.06, 100 * knn.test(car.getInstanceList()).getErrorRate(), 0.01);
        knn.train(tictactoe.getInstanceList(), knnParameter);
        knn.saveTxt("models/knn-tictactoe.txt");
        knn.loadModel("models/knn-tictactoe.txt");
        assertEquals(32.57, 100 * knn.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
    }

}