package Classification.Classifier;

import Classification.Model.Parametric.NaiveBayesModel;
import org.junit.Test;

import static org.junit.Assert.*;

public class NaiveBayesTest extends ClassifierTest{

    @Test
    public void testTrain() {
        NaiveBayesModel naiveBayes = new NaiveBayesModel();
        naiveBayes.train(iris.getInstanceList(), null);
        naiveBayes.saveTxt("models/naiveBayes-iris.txt");
        naiveBayes.loadModel("models/naiveBayes-iris.txt");
        assertEquals(5.33, 100 * naiveBayes.test(iris.getInstanceList()).getErrorRate(), 0.01);
        naiveBayes.train(bupa.getInstanceList(), null);
        naiveBayes.saveTxt("models/naiveBayes-bupa.txt");
        naiveBayes.loadModel("models/naiveBayes-bupa.txt");
        assertEquals(38.55, 100 * naiveBayes.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        naiveBayes.train(dermatology.getInstanceList(), null);
        naiveBayes.saveTxt("models/naiveBayes-dermatology.txt");
        naiveBayes.loadModel("models/naiveBayes-dermatology.txt");
        assertEquals(9.56, 100 * naiveBayes.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        naiveBayes.train(car.getInstanceList(), null);
        assertEquals(12.91, 100 * naiveBayes.test(car.getInstanceList()).getErrorRate(), 0.01);
        naiveBayes.train(tictactoe.getInstanceList(), null);
        assertEquals(30.17, 100 * naiveBayes.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
    }

}