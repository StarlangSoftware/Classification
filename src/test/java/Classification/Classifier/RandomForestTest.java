package Classification.Classifier;

import Classification.Parameter.RandomForestParameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class RandomForestTest extends ClassifierTest{

    @Test
    public void testTrain1() {
        RandomForest randomForest = new RandomForest();
        RandomForestParameter randomForestParameter = new RandomForestParameter(1, 100, 35);
        randomForest.train(iris.getInstanceList(), randomForestParameter);
        randomForest.getModel().saveTxt("models/randomforest-iris.txt");
        randomForest.loadModel("models/randomforest-iris.txt");
        assertEquals(0.0, 100 * randomForest.test(iris.getInstanceList()).getErrorRate(), 0.01);
    }

    @Test
    public void testTrain2() {
        RandomForest randomForest = new RandomForest();
        RandomForestParameter randomForestParameter = new RandomForestParameter(1, 100, 35);
        randomForest.train(bupa.getInstanceList(), randomForestParameter);
        randomForest.getModel().saveTxt("models/randomforest-bupa.txt");
        randomForest.loadModel("models/randomforest-bupa.txt");
        assertEquals(0.0, 100 * randomForest.test(bupa.getInstanceList()).getErrorRate(), 0.01);
    }

    @Test
    public void testTrain3() {
        RandomForest randomForest = new RandomForest();
        RandomForestParameter randomForestParameter = new RandomForestParameter(1, 100, 35);
        randomForest.train(dermatology.getInstanceList(), randomForestParameter);
        randomForest.getModel().saveTxt("models/randomforest-dermatology.txt");
        randomForest.loadModel("models/randomforest-dermatology.txt");
        assertEquals(0.0, 100 * randomForest.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
    }

    @Test
    public void testTrain4() {
        RandomForest randomForest = new RandomForest();
        RandomForestParameter randomForestParameter = new RandomForestParameter(1, 100, 35);
        randomForest.train(car.getInstanceList(), randomForestParameter);
        randomForest.getModel().saveTxt("models/randomforest-car.txt");
        randomForest.loadModel("models/randomforest-car.txt");
        assertEquals(0.0, 100 * randomForest.test(car.getInstanceList()).getErrorRate(), 0.01);
    }

    @Test
    public void testTrain5() {
        RandomForest randomForest = new RandomForest();
        RandomForestParameter randomForestParameter = new RandomForestParameter(1, 100, 35);
        randomForest.train(tictactoe.getInstanceList(), randomForestParameter);
        randomForest.getModel().saveTxt("models/randomforest-tictactoe.txt");
        randomForest.loadModel("models/randomforest-tictactoe.txt");
        assertEquals(0.0, 100 * randomForest.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
    }

    @Test
    public void testTrain6() {
        RandomForest randomForest = new RandomForest();
        RandomForestParameter randomForestParameter = new RandomForestParameter(1, 100, 35);
        randomForest.train(carIndexed.getInstanceList(), randomForestParameter);
        randomForest.getModel().saveTxt("models/randomforest-carIndexed.txt");
        randomForest.loadModel("models/randomforest-carIndexed.txt");
        assertEquals(0.0, 100 * randomForest.test(carIndexed.getInstanceList()).getErrorRate(), 0.01);
    }

}