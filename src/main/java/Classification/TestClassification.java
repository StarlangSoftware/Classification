package Classification;

import Classification.Attribute.AttributeType;
import Classification.Classifier.*;
import Classification.DataSet.DataDefinition;
import Classification.DataSet.DataSet;
import Classification.Experiment.Experiment;
import Classification.Experiment.KFoldRun;
import Classification.Experiment.StratifiedKFoldRun;
import Classification.Filter.Normalize;
import Classification.Model.Svm.KernelType;
import Classification.Parameter.*;
import Classification.Performance.ExperimentPerformance;

import java.io.File;
import java.util.ArrayList;

public class TestClassification {

    private static Parameter defaultParameter() { return new Parameter(1);}

    private static Parameter linearPerceptron(){
        return new LinearPerceptronParameter(1, 0.1, 0.95, 0.2, 10);
    }

    private static Parameter multiLayerPerceptron(){
        return new MultiLayerPerceptronParameter(1, 0.1, 0.95, 0.2, 30, 5);
    }

    private static Parameter deepNetwork(){
        ArrayList hiddens = new ArrayList();
        hiddens.add(3);
        hiddens.add(5);
        return new DeepNetworkParameter(1, 0.5, 0.95, 0.2, 30, hiddens);
    }

    private static Parameter svm(){
        return new SvmParameter(1, KernelType.RBF, 1, 0.01, 0, 1);
    }

    private static DataSet readIris(){
        ArrayList<AttributeType> attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 4; i++){
            attributeTypes.add(AttributeType.CONTINUOUS);
        }
        DataDefinition dataDefinition = new DataDefinition(attributeTypes);
        return new DataSet(dataDefinition, ",", "Data/Classification/iris.data");
    }

    private static DataSet readBupa(){
        ArrayList<AttributeType> attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 6; i++){
            attributeTypes.add(AttributeType.CONTINUOUS);
        }
        DataDefinition dataDefinition = new DataDefinition(attributeTypes);
        return new DataSet(dataDefinition, ",", "Data/Classification/bupa.data");
    }

    private static DataSet readDermatology(){
        ArrayList<AttributeType> attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 34; i++){
            attributeTypes.add(AttributeType.CONTINUOUS);
        }
        DataDefinition dataDefinition = new DataDefinition(attributeTypes);
        return new DataSet(dataDefinition, ",", "Data/Classification/dermatology.data");
    }

    private static DataSet readRingnorm(){
        ArrayList<AttributeType> attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 20; i++){
            attributeTypes.add(AttributeType.CONTINUOUS);
        }
        DataDefinition dataDefinition = new DataDefinition(attributeTypes);
        return new DataSet(dataDefinition, "\\s+", "Data/Classification/ringnorm.data");
    }

    private static DataSet readTwonorm(){
        ArrayList<AttributeType> attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 20; i++){
            attributeTypes.add(AttributeType.CONTINUOUS);
        }
        DataDefinition dataDefinition = new DataDefinition(attributeTypes);
        return new DataSet(dataDefinition, "\\s+", "Data/Classification/twonorm.data");
    }

    private static DataSet readCar(){
        ArrayList<AttributeType> attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 6; i++){
            attributeTypes.add(AttributeType.DISCRETE);
        }
        DataDefinition dataDefinition = new DataDefinition(attributeTypes);
        return new DataSet(dataDefinition, ",", "Data/Classification/car.data");
    }

    private static DataSet readNursery(){
        ArrayList<AttributeType> attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 8; i++){
            attributeTypes.add(AttributeType.DISCRETE);
        }
        DataDefinition dataDefinition = new DataDefinition(attributeTypes);
        return new DataSet(dataDefinition, ",", "Data/Classification/nursery.data");
    }

    private static DataSet readConnect4(){
        ArrayList<AttributeType> attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 42; i++){
            attributeTypes.add(AttributeType.DISCRETE);
        }
        DataDefinition dataDefinition = new DataDefinition(attributeTypes);
        return new DataSet(dataDefinition, ",", "Data/Classification/connect4.data");
    }

    private static DataSet readTicTacToe(){
        ArrayList<AttributeType> attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 9; i++){
            attributeTypes.add(AttributeType.DISCRETE);
        }
        DataDefinition dataDefinition = new DataDefinition(attributeTypes);
        return new DataSet(dataDefinition, ",", "Data/Classification/tictactoe.data");
    }

    private static DataSet readChess(){
        ArrayList<AttributeType> attributeTypes = new ArrayList<AttributeType>();
        for (int i = 0; i < 6; i++){
            if (i % 2 == 0){
                attributeTypes.add(AttributeType.DISCRETE);
            } else {
                attributeTypes.add(AttributeType.CONTINUOUS);
            }
        }
        DataDefinition dataDefinition = new DataDefinition(attributeTypes);
        return new DataSet(dataDefinition, ",", "Data/Classification/chess.data");
    }

    private static void testAutoEncoder(){
        DataSet dataSet = readIris();
        Parameter parameter = multiLayerPerceptron();
        KFoldRun run = new KFoldRun(10);
        ExperimentPerformance results = null;
        try {
            results = run.execute(new Experiment(new AutoEncoder(), parameter, dataSet));
            System.out.println(results.meanPerformance().getErrorRate());
        } catch (DiscreteFeaturesNotAllowed discreteFeaturesNotAllowed) {
            discreteFeaturesNotAllowed.printStackTrace();
        }
    }

    private static void testNlp(){
        DataSet dataSet = new DataSet(new File("shallowparse.txt"));
        KFoldRun run = new KFoldRun(10);
        try {
            ExperimentPerformance results = run.execute(new Experiment(new RandomClassifier(), new Parameter(10), dataSet));
            System.out.println(100 * results.meanClassificationPerformance().getErrorRate() + " " + 100 * results.standardDeviationClassificationPerformance().getErrorRate());
        } catch (DiscreteFeaturesNotAllowed discreteFeaturesNotAllowed) {
            discreteFeaturesNotAllowed.printStackTrace();
        }
    }

    private static void test(){
        DataSet dataSet = readDermatology();
/*        DiscreteToContinuous discreteToContinuous = new DiscreteToContinuous(dataSet);
        discreteToContinuous.convert();*/
/*        DiscreteToIndexed discreteToIndexed = new DiscreteToIndexed(dataSet);
        discreteToIndexed.convert();*/
/*        LaryToBinary laryToBinary = new LaryToBinary(dataSet);
        laryToBinary.convert();*/
/*        Pca pca = new Pca(dataSet, 2);
        pca.convert();*/
/*        Parameter parameter = defaultParameter();*/
        Normalize normalize = new Normalize(dataSet);
        normalize.convert();
        Parameter parameter = svm();
        StratifiedKFoldRun run = new StratifiedKFoldRun(10);
        ExperimentPerformance results = null;
        try {
            results = run.execute(new Experiment(new Svm(), parameter, dataSet));
            System.out.println(100 * results.meanClassificationPerformance().getErrorRate() + " " + 100 * results.standardDeviationClassificationPerformance().getErrorRate());
        } catch (DiscreteFeaturesNotAllowed discreteFeaturesNotAllowed) {
            discreteFeaturesNotAllowed.printStackTrace();
        }
    }

    public static void main(String[] args){
        testNlp();
    }
}
